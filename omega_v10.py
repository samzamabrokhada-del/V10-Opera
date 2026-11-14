#!/usr/bin/env python3
"""PRO-PITCH OMEGA V10 | open, offline quantitative football research engine.

Requires numpy, scipy, scikit-learn. Usage:
  python omega_v10.py backtest --matches data/matches.csv --out reports/run
  python omega_v10.py analyze --history data/matches.csv --fixtures data/upcoming.csv --out reports/preview

Never submits bets, connects to a bookmaker, or invents missing xG/lineup/odds.
Quoted-edge decisions require sufficient PREVIOUS-SEASON calibration and actual
captured odds; exploratory confidence-ranked shortlists are not betting advice.
"""
from __future__ import annotations
import argparse,csv, json, math, hashlib
from dataclasses import dataclass, asdict
from datetime import datetime,timezone,timedelta
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.linear_model import LogisticRegression
import omega_v8_core as core

UTC=timezone.utc
MARKETS=core.MARKETS
CONTEXT_FIELDS=('rest_gap','travel_gap','missing_gap','motivation_gap','depth_gap','timezone_gap','congestion_gap')


def dt(x):return core.parse_dt(str(x))
def iso(x):return x.astimezone(UTC).isoformat()
def fnum(x):return float(x) if str(x).strip() not in ('','None','nan','NA') else None

def date_key(r):return r['kickoff_dt'].date().isoformat()
def match_key(r):return (r['league'],r['kickoff_dt'],r['home'],r['away'])

def load_matches(paths, require_score=True):
    rows=[]
    for path in paths:
      with open(path,encoding='utf-8-sig',newline='') as handle:
       reader=csv.DictReader(handle)
       needed={'league','season','kickoff','home','away'} | ({'home_goals','away_goals'} if require_score else set())
       if not needed.issubset(reader.fieldnames or []): raise ValueError(f'{path}: missing {needed-set(reader.fieldnames or [])}')
       for line,r in enumerate(reader,2):
        item={k:r[k].strip() for k in ('league','season','home','away')};item['kickoff_dt']=dt(r['kickoff'])
        if any(not item[k] for k in ('league','season','home','away')) or item['home']==item['away']:raise ValueError(f'{path}:{line}: invalid fixture')
        if require_score or str(r.get('home_goals','')).strip():
         item['hg']=int(r['home_goals']);item['ag']=int(r['away_goals'])
         if min(item['hg'],item['ag'])<0:raise ValueError(f'{path}:{line}: negative goals')
        else:item['hg']=item['ag']=None
        for name in ('home_xg','away_xg','home_npxg','away_npxg','venue_lat','venue_lon'):
         v=fnum(r.get(name,''));item[name]=v
         if v is not None and (not math.isfinite(v) or (name.endswith('xg') and v<0)):
          raise ValueError(f'{path}:{line}: invalid {name}')
        if (item['venue_lat'] is None)!=(item['venue_lon'] is None):raise ValueError('Venue requires both latitude and longitude')
        if item['venue_lat'] is not None and not(-90<=item['venue_lat']<=90 and -180<=item['venue_lon']<=180):raise ValueError('Invalid coordinates')
        item['timing_verified']=str(r.get('timing_verified','')).strip().lower() in ('1','true','yes','verified')
        supplied=(r.get('result_available_at') or '').strip()
        item['result_available_dt']=dt(supplied) if supplied else item['kickoff_dt']+timedelta(hours=3)
        item['result_timing_verified']=bool(supplied)
        if supplied and item['result_available_dt']<=item['kickoff_dt']:raise ValueError('Result availability must be after kickoff')
        item['file']=str(path); item['line']=line
        rows.append(item)
    rows.sort(key=lambda r:(r['kickoff_dt'],r['league'],r['home']))
    ids=[match_key(r) for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate match identity')
    return rows


def load_context(path):
    if not path:return {}
    out=defaultdict(list)
    with open(path,encoding='utf-8-sig',newline='') as h:
     d=csv.DictReader(h)
     need={'league','kickoff','home','away','recorded_at'}
     if not need.issubset(d.fieldnames or []): raise ValueError(f'Context missing {need-set(d.fieldnames or [])}')
     for n,r in enumerate(d,2):
      ko=dt(r['kickoff']);at=dt(r['recorded_at'])
      if at>=ko:raise ValueError(f'Context at/after kickoff (line {n})')
      x={'recorded_at':at,'source':r.get('source','')}
      for name in ('home_missing_starters','away_missing_starters'):
       v=fnum(r.get(name,'')); x[name]=v
       if v is not None and not (0<=v<=11):raise ValueError(f'Bad {name} line {n}')
      for name in ('home_squad_depth','away_squad_depth'):
       v=fnum(r.get(name,''));x[name]=v
       if v is not None and not (0<=v<=100):raise ValueError(f'Bad {name} line {n}')
      for name in ('home_timezone_crossings','away_timezone_crossings'):
       v=fnum(r.get(name,''));x[name]=v
       if v is not None and not (0<=v<=12):raise ValueError(f'Bad {name} line {n}')
      for name in ('home_motivation','away_motivation'):
       v=fnum(r.get(name,''));x[name]=v
       if v is not None and not (-1<=v<=1):raise ValueError(f'Bad {name} line {n}')
      out[(r['league'].strip(),ko,r['home'].strip(),r['away'].strip())].append(x)
    for v in out.values():v.sort(key=lambda z:z['recorded_at'])
    return out


def context_asof(store, fixture, at):
    records=store.get(match_key(fixture),[])
    valid=[x for x in records if x['recorded_at']<=at]
    return valid[-1] if valid else None


def load_odds(path):
    if not path:return {}
    rows=defaultdict(list)
    with open(path,encoding='utf-8-sig',newline='') as h:
     d=csv.DictReader(h)
     need={'league','kickoff','home','away','market','decimal_odds','bookmaker','recorded_at'}
     if not need.issubset(d.fieldnames or []):raise ValueError(f'Odds missing {need-set(d.fieldnames or [])}')
     for n,r in enumerate(d,2):
      market=r['market'].strip(); ko=dt(r['kickoff']);at=dt(r['recorded_at']); o=float(r['decimal_odds'])
      if market not in MARKETS or not math.isfinite(o) or o<=1 or not r['bookmaker'].strip():raise ValueError(f'Invalid odds at line {n}')
      if at>=ko:raise ValueError(f'Post-kickoff odds at line {n}')
      key=(r['league'].strip(),ko,r['home'].strip(),r['away'].strip())
      rows[key].append({'market':market,'bookmaker':r['bookmaker'].strip(),'odds':o,'recorded_at':at,'source':r.get('source','')})
    for v in rows.values():v.sort(key=lambda r:r['recorded_at'])
    return rows


def km(p1,p2):
    la1,lo1=map(math.radians,p1);la2,lo2=map(math.radians,p2)
    a=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 6371*2*math.asin(min(1,math.sqrt(a)))


def features(fixture, earlier_schedule, context_store, decision_at):
    """Use only earlier fixtures and pre-decision snapshots; no result values consumed."""
    prev={}
    recent_14=Counter()
    for r in earlier_schedule:
      if r['kickoff_dt']>=decision_at or r['league']!=fixture['league']:continue
      for t in ('home','away'):
        prev[r[t]]=r
        if (decision_at-r['kickoff_dt']).total_seconds()<=14*86400: recent_14[r[t]]+=1
    data=[];known={}
    for side in ('home','away'):
      last=prev.get(fixture[side]);rest=None;travel=None
      if last:
       rest=max(0.,(fixture['kickoff_dt']-last['kickoff_dt']).total_seconds()/86400)
       if all(last.get(k) is not None and fixture.get(k) is not None for k in ('venue_lat','venue_lon')):
        travel=km((last['venue_lat'],last['venue_lon']),(fixture['venue_lat'],fixture['venue_lon']))
      known[side+'_rest_days']=None if rest is None else round(rest,2)
      known[side+'_travel_km']=None if travel is None else round(travel,1)
    snapshot=context_asof(context_store,fixture,decision_at)
    def present(k):return snapshot.get(k) if snapshot is not None else None
    def difference(a,b,scale,cap):
      return float(np.clip((a-b)/scale,-cap,cap)) if a is not None and b is not None else 0.
    # away better rest -> stronger away, while away more travel/missing -> stronger home
    data=[
      difference(known['home_rest_days'],known['away_rest_days'],7,2),
      difference(known['away_travel_km'],known['home_travel_km'],1500,3),
      difference(present('away_missing_starters'),present('home_missing_starters'),3,3),
      difference(present('home_motivation'),present('away_motivation'),1,2),
      difference(present('home_squad_depth'),present('away_squad_depth'),30,2),
      difference(present('away_timezone_crossings'),present('home_timezone_crossings'),4,2),
      float(np.clip((recent_14[fixture['away']]-recent_14[fixture['home']])/3,-2,2))]

    known['captured_context']=bool(snapshot);known['home_matches_last_14d']=recent_14[fixture['home']];known['away_matches_last_14d']=recent_14[fixture['away']];known['rest_data_available']=known['home_rest_days'] is not None and known['away_rest_days'] is not None
    known['travel_data_available']=known['home_travel_km'] is not None and known['away_travel_km'] is not None
    known['squad_data_available']=present('home_missing_starters') is not None and present('away_missing_starters') is not None
    known['motivation_data_available']=present('home_motivation') is not None and present('away_motivation') is not None
    known['squad_depth_available']=present('home_squad_depth') is not None and present('away_squad_depth') is not None
    known['timezone_crossing_available']=present('home_timezone_crossings') is not None and present('away_timezone_crossings') is not None
    return np.asarray(data,dtype=float),known


@dataclass
class GoalModel:
    intercept:float
    home_advantage:float
    rho:float
    attack:dict
    defence:dict
    context_beta:tuple
    trained_matches:int
    xg_training_matches:int
    asof:str
    league:str
    def rates(self,home,away,f):
      f=np.asarray(f,dtype=float)
      adjust=float(np.dot(np.asarray(self.context_beta),f))
      lh=math.exp(float(np.clip(self.intercept+self.home_advantage+self.attack.get(home,0.)-self.defence.get(away,0.)+adjust,-3.2,2.4)))
      la=math.exp(float(np.clip(self.intercept+self.attack.get(away,0.)-self.defence.get(home,0.)-adjust,-3.2,2.4)))
      return lh,la


def fit_goals(history, asof, league, context_store=None, half_life=240.,xg_strength=.35,min_history=30):
    """Regularised weighted Dixon–Coles MLE + optional auxiliary observed xG targets.

    Auxiliary xG is a continuous likelihood *penalty*, not a pretend goal count.
    Context coefficients are fitted, never assigned an unsupported fixed advantage.
    """
    context_store=context_store or {}
    h=[r for r in history if r['league']==league and r['hg'] is not None and r['result_available_dt']<asof and 0<(asof-r['kickoff_dt']).total_seconds()<3.5*366*86400]
    if len(h)<min_history:return None
    t=sorted({r['home'] for r in h}|{r['away'] for r in h});n=len(t);idx={a:i for i,a in enumerate(t)}
    home_ix=np.array([idx[r['home']] for r in h]); away_ix=np.array([idx[r['away']] for r in h])
    hg=np.array([r['hg'] for r in h],dtype=float);ag=np.array([r['ag'] for r in h],dtype=float)
    age=np.array([(asof-r['kickoff_dt']).total_seconds()/86400 for r in h]);w=np.exp(-math.log(2)*age/half_life)
    # npXG measures chance quality excluding penalties. Where both xG and npXG
    # exist, estimate each league's historical penalty contribution using only
    # training history and blend the penalty-normalized npXG with xG. Where
    # both are unavailable, NO feature is fabricated and only goals are fitted.
    pairs=[]
    for r in h:
      for side in ('home','away'):
        a,b=r[side+'_xg'],r[side+'_npxg']
        if a is not None and b is not None:pairs.append(max(0.,a-b))
    penalty_baseline=float(np.median(pairs)) if len(pairs)>=15 else None
    def auxiliary(row,side):
      a,b=row[side+'_xg'],row[side+'_npxg']
      if a is not None and b is not None and penalty_baseline is not None:return .5*a+.5*(b+penalty_baseline)
      if a is not None:return a
      if b is not None and penalty_baseline is not None:return b+penalty_baseline
      return float('nan')
    xgh=np.array([auxiliary(r,'home') for r in h]);xga=np.array([auxiliary(r,'away') for r in h]);valid=np.isfinite(xgh)&np.isfinite(xga)
    # Pre-match context for *historical* matches: never source the context from later snapshots.
    known_schedule=sorted(h,key=lambda r:r['kickoff_dt']);Z=[]
    for r in h:
      f,_=features(r,[p for p in known_schedule if p['kickoff_dt']<r['kickoff_dt']],context_store,r['kickoff_dt']-timedelta(minutes=60));Z.append(f)
    Z=np.asarray(Z)
    # A context feature cannot learn if it is missing/constant in the available history.
    active=np.where(np.std(Z,axis=0)>0.10)[0];Zc=Z[:,active] if len(active) else np.empty((len(h),0))
    logfac=gammaln(hg+1)+gammaln(ag+1)
    mu0=max(.35,float(np.average(hg+ag,weights=w)/2))
    # Params: log mean, home edge, low-score correction, team attack/defence, contexts.
    start=np.zeros(3+2*n+len(active));start[0]=math.log(mu0);start[1]=.15;start[2]=-.02
    bounds=[(-1.6,1.5),(-.5,.8),(-.10,.10)]+[(-1.6,1.6)]*(2*n)+[(-.32,.32)]*len(active)
    def cost(x):
      att=x[3:3+n];de=x[3+n:3+2*n];att=att-att.mean();de=de-de.mean()
      adj=Zc@x[3+2*n:] if len(active) else np.zeros(len(h))
      lh=np.exp(np.clip(x[0]+x[1]+att[home_ix]-de[away_ix]+adj,-3.2,2.4))
      la=np.exp(np.clip(x[0]+att[away_ix]-de[home_ix]-adj,-3.2,2.4))
      tau=np.ones(len(h))
      for a,b in ((0,0),(0,1),(1,0),(1,1)):
       loc=(hg==a)&(ag==b)
       if loc.any():tau[loc]=core.dc_tau(a,b,lh[loc],la[loc],x[2])
      goals=hg*np.log(lh)-lh+ag*np.log(la)-la-logfac+np.log(np.maximum(tau,1e-10))
      penalty=1.7*(att@att+de@de)+.5*x[1]**2+.05*(x[2]/.1)**2+10.*(x[3+2*n:]@x[3+2*n:])
      if valid.any() and xg_strength>0:
       # Gaussian xG observation model with large noise scale; not assumed to be independent goals.
       ex=0.5*(np.square((xgh[valid]-lh[valid])/1.25)+np.square((xga[valid]-la[valid])/1.25))
       penalty+=xg_strength*float(w[valid]@ex)
      return float(-(w@goals)+penalty)
    opt=minimize(cost,start,bounds=bounds,method='L-BFGS-B',options={'maxiter':110,'ftol':2e-6})
    vec=opt.x;att=vec[3:3+n]-vec[3:3+n].mean();de=vec[3+n:3+2*n]-vec[3+n:3+2*n].mean()
    context_beta=np.zeros(len(CONTEXT_FIELDS))
    for i,j in enumerate(active):context_beta[j]=vec[3+2*n+i]
    return GoalModel(float(vec[0]),float(vec[1]),float(vec[2]),{team:float(att[i]) for team,i in idx.items()},{team:float(de[i]) for team,i in idx.items()},tuple(float(v) for v in context_beta),len(h),int(valid.sum()),iso(asof),league)


def settle(m,hg,ag):return core.settle(m,hg,ag)

def predict(model, row, z, calibrators=None):
    h,a=model.rates(row['home'],row['away'],z)
    mat=core.score_matrix(h,a,model.rho)
    if calibrators:mat=calibrate_matrix(mat,calibrators)
    probs=core.model_probs(mat)
    return {'lambda_home':round(h,6),'lambda_away':round(a,6),'probs':probs,'matrix':mat}


def train_calibrators(previous, min_n=90):
    """Only previously completed seasons are accepted by the caller."""
    if len(previous)<min_n:return {}
    out={}
    for market in MARKETS:
      xs=np.array([r['raw_probs'][market] for r in previous],dtype=float)
      ys=np.array([settle(market,r['hg'],r['ag']) for r in previous],dtype=int)
      if len(xs)<min_n or min(ys.sum(),len(ys)-ys.sum())<8 or np.std(xs)<.005:continue
      eps=1e-5; x=np.log(np.clip(xs,eps,1-eps)/(1-np.clip(xs,eps,1-eps))).reshape(-1,1)
      clf=LogisticRegression(C=1.0,max_iter=300).fit(x,ys)
      out[market]=(float(clf.coef_[0,0]),float(clf.intercept_[0]),len(xs))
    return out


def calibrate_matrix(matrix,calibrators):
    """Soft KL-like iterative reweighting. Final markets always come from ONE matrix.

    Independent binary targets cannot always be satisfied exactly; do not report
    direct calibrator outputs as mutually coherent market probabilities.
    """
    base=core.model_probs(matrix)
    targets={}
    for m,(k,b,n) in calibrators.items():
      p=float(np.clip(base[m],1e-5,1-1e-5));logit=math.log(p/(1-p))
      q=float(1/(1+math.exp(-np.clip(k*logit+b,-20,20))))
      targets[m]=float(np.clip(q,1e-4,1-1e-4))
    result=matrix.copy()
    for _ in range(7):
      for m,q in targets.items():
       sel=core.mask(m,len(result));p=float(result[sel].sum())
       if p<1e-8 or p>1-1e-8:continue
       power=.10 # shrink contradictory constraints rather than force arbitrary probabilities
       result[sel]*=float((q/p)**power)
       result[~sel]*=float(((1-q)/(1-p))**power)
       result/=result.sum()
    return result


def quote_at(quotes, match, decision_at, max_age_hours=36):
    """Most recent per-bookmaker quote as known at decision time."""
    candidates={}
    for q in quotes.get(match_key(match),[]):
      if q['recorded_at']>decision_at or (decision_at-q['recorded_at']).total_seconds()>max_age_hours*3600:continue
      key=(q['market'],q['bookmaker'])
      if key not in candidates or q['recorded_at']>candidates[key]['recorded_at']:
       candidates[key]=q
    best={}
    for q in candidates.values():
      if q['market'] not in best or q['odds']>best[q['market']]['odds']:best[q['market']]=q
    return best


def clv_for(quotes, match, winning_quote):
    # compare with same bookmaker's last strictly PRE-KICKOFF quote
    chain=[q for q in quotes.get(match_key(match),[]) if q['market']==winning_quote['market'] and q['bookmaker']==winning_quote['bookmaker'] and q['recorded_at']>=winning_quote['recorded_at'] and q['recorded_at']<match['kickoff_dt']]
    if len(chain)<2:return None
    last=max(chain,key=lambda q:q['recorded_at'])
    return (winning_quote['odds']/last['odds']-1) if last['recorded_at']>winning_quote['recorded_at'] else None


def effective_calibration_n(calibrators,market):return calibrators.get(market,(0,0,0))[2]

def edge_candidates(row, probs,calibrators,quotes,decision_at,min_ev=.04,uncertainty_multiplier=.5):
    available=quote_at(quotes,row,decision_at)
    out=[]
    for market,q in available.items():
     if market not in calibrators:continue
     n=effective_calibration_n(calibrators,market)
     p=probs[market];uncertainty=math.sqrt(p*(1-p)/max(1,n))*uncertainty_multiplier
     conservative=max(.001,p-uncertainty)
     edge=conservative*q['odds']-1
     if edge>=min_ev:out.append({'market':market,'odds':q['odds'],'bookmaker':q['bookmaker'],'recorded_at':iso(q['recorded_at']),'p':p,'conservative_p':conservative,'edge':edge,'clv':clv_for(quotes,row,q)})
    return sorted(out,key=lambda x:-x['edge'])


def ece(values,bins=8):
    if not values:return None
    n=len(values);tot=0.
    for b in range(bins):
     subset=[(p,y) for p,y in values if (b/bins<=p<(b+1)/bins) or (b==bins-1 and p==1)]
     if subset:tot+=len(subset)/n*abs(np.mean([x[0] for x in subset])-np.mean([x[1] for x in subset]))
    return round(tot,5)


def evaluate(rows, paper, diagnostics, priced, skipped,conf):
    out={'inputs':len(rows),'forecasted':len(paper),'skipped':skipped,'diagnostic':{},'priced':{},'calibration':{},'timing_verified_count':sum(bool(x['timing_verified']) for x in rows),'verified_result_availability_count':sum(bool(x['result_timing_verified']) for x in rows)}
    if paper:
     actual_goals=np.mean([r['hg']+r['ag'] for r in paper]);mean_pred=np.mean([r['lambda_home']+r['lambda_away'] for r in paper]);
     out['mean_goals_predicted']=round(float(mean_pred),4);out['mean_goals_actual']=round(float(actual_goals),4);out['mean_goal_error_pct']=round(100*(mean_pred/actual_goals-1),2) if actual_goals else None
     out['one_x_two_log_loss']=round(float(-np.mean([math.log(max(1e-12,r['probs'][('Home win' if r['hg']>r['ag'] else 'Draw' if r['hg']==r['ag'] else 'Away win')])) for r in paper])),4)
     out['one_x_two_brier']=round(float(np.mean([sum((r['probs'][m]-int(settle(m,r['hg'],r['ag'])))**2 for m in ('Home win','Draw','Away win')) for r in paper])),4)
     for m in ('1X','X2','Over 1.5','Under 3.5','BTTS Yes'):
      y=[(r['probs'][m],int(settle(m,r['hg'],r['ag']))) for r in paper]
      out['calibration'][m]={'estimated_mean':round(float(np.mean([a for a,b in y])),4),'hit_rate':round(float(np.mean([b for a,b in y])),4),'ece':ece(y),'n':len(y)}
    else:out['mean_goals_predicted']=out['mean_goals_actual']=out['mean_goal_error_pct']=out['one_x_two_log_loss']=out['one_x_two_brier']=None
    for name,picks in [('research_top5',diagnostics),('odds_qualified',priced)]:
     n=len(picks);wins=sum(int(x.get('won',False)) for x in picks)
     obj={'picks':n,'wins':wins,'losses':n-wins,'hit_rate':round(wins/n,4) if n else None}
     if n:
      obj['baseline_always_1X_on_picks']=round(sum(r['hg']>=r['ag'] for r in picks)/n,4)
      obj['baseline_over_1_5_on_picks']=round(sum(r['hg']+r['ag']>=2 for r in picks)/n,4)
      obj['dates_with_picks']=len(set(x['date'] for x in picks))
     if name=='odds_qualified':
      obj['yield_flat_stake']=round(sum(((r['odds']-1) if r['won'] else -1) for r in picks)/n,5) if n else None
      obj['average_clv']=round(float(np.mean([r['clv'] for r in picks if r['clv'] is not None])),5) if any(r['clv'] is not None for r in picks) else None
     out['diagnostic' if name=='research_top5' else 'priced']=obj
    out['research_only']=not conf['strict_timestamp_verification'] or not priced
    out['caveats']=['This is retrospective research, not proof of a betting edge.','Dataset provenance and independent kickoff verification are required for strict audits.','Market probabilities are not verified bookmaker probabilities.','No automatic wagering or cash connection.']
    return out


def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows and not fields:fields=['no_rows']
    if fields is None:fields=list(rows[0])
    with path.open('w',encoding='utf-8',newline='') as f:
     writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def run_backtest(matches,context_store=None,odds=None,out=None,threshold=.80,max_daily=5,min_ev=.04,fit_days=28,warmup=35,decision_minutes=60):
    context_store=context_store or {};odds=odds or {};allrows=sorted(matches,key=lambda x:x['kickoff_dt'])
    prior_raw=[];season_calibration={};trained_cache={};predictions=[];skipped=Counter()
    for r in allrows:
     asof=r['kickoff_dt']-timedelta(minutes=decision_minutes)
     past=[x for x in allrows if x['league']==r['league'] and x['result_available_dt']<asof and x['kickoff_dt']<r['kickoff_dt']]
     if len(past)<warmup:skipped['warmup']+=1;continue
     cache=trained_cache.get(r['league'])
     if cache is None or (asof-cache['asof']).total_seconds()>fit_days*86400:
      model=fit_goals(past,asof,r['league'],context_store,min_history=warmup)
      if model is None:skipped['no_training']+=1;continue
      trained_cache[r['league']]={'asof':asof,'model':model}
     else:model=cache['model']
     season_key=(r['league'],r['season'])
     if season_key not in season_calibration:
      old=[p for p in prior_raw if p['league']==r['league'] and p['season']!=r['season'] and p['result_available_dt']<asof]
      season_calibration[season_key]=train_calibrators(old)
     calibrators=season_calibration[season_key]
     z,context_note=features(r,[x for x in allrows if x['kickoff_dt']<asof],context_store,asof)
     y=predict(model,r,z,calibrators)
     raw=predict(model,r,z)
     dc='1X' if y['probs']['1X']>=y['probs']['X2'] else 'X2'
     odds_candidates=edge_candidates(r,y['probs'],calibrators,odds,asof,min_ev=min_ev)
     entry={'league':r['league'],'season':r['season'],'date':date_key(r),'kickoff':iso(r['kickoff_dt']),'home':r['home'],'away':r['away'],'hg':r['hg'],'ag':r['ag'],'lambda_home':y['lambda_home'],'lambda_away':y['lambda_away'],'raw_probs':raw['probs'],'probs':y['probs'],'dc_market':dc,'dc_p':round(y['probs'][dc],5),'calibrated_market_count':len(calibrators),'xg_training_matches':model.xg_training_matches,'context':context_note,'research_win':settle(dc,r['hg'],r['ag']),'odds_candidates':odds_candidates,'timing_verified':r['timing_verified'],'result_timing_verified':r['result_timing_verified']}
     predictions.append(entry)
     prior_raw.append({'league':r['league'],'season':r['season'],'raw_probs':raw['probs'],'hg':r['hg'],'ag':r['ag'],'result_available_dt':r['result_available_dt']})
    by_day=defaultdict(list)
    for r in predictions:by_day[r['date']].append(r)
    diagnostic=[];priced=[]
    for day,group in sorted(by_day.items()):
     qual=sorted([r for r in group if r['dc_p']>=threshold],key=lambda r:(-r['dc_p'],r['league'],r['home'],r['away']))[:max_daily]
     for r in qual:
      diagnostic.append({'date':day,'league':r['league'],'home':r['home'],'away':r['away'],'hg':r['hg'],'ag':r['ag'],'market':r['dc_market'],'probability':r['dc_p'],'won':r['research_win'],'calibrated':r['calibrated_market_count']>0})
     pool=[]
     for r in group:
      for opt in r['odds_candidates']:
       if not r['timing_verified'] or not r['result_timing_verified']:continue
       pool.append((opt['edge'],r,opt))
     seen=set()
     for edge,r,opt in sorted(pool,key=lambda t:-t[0]):
      identity=(r['league'],r['kickoff'],r['home'],r['away'])
      if identity in seen:continue
      seen.add(identity)
      priced.append({'date':day,'league':r['league'],'home':r['home'],'away':r['away'],'hg':r['hg'],'ag':r['ag'],'market':opt['market'],'probability':round(opt['p'],5),'odds':opt['odds'],'edge':round(opt['edge'],5),'bookmaker':opt['bookmaker'],'clv':opt['clv'],'won':settle(opt['market'],r['hg'],r['ag'])})
      if len(seen)>=max_daily:break
    report=evaluate(allrows,predictions,diagnostic,priced,dict(skipped),{'strict_timestamp_verification':all(x['timing_verified'] and x['result_timing_verified'] for x in allrows)})
    report['filters']={'research_double_chance_min_p':threshold,'min_conservative_expected_value':min_ev,'global_max_daily':max_daily,'fit_cadence_days':fit_days,'warmup_matches_by_league':warmup,'decision_minutes_before_kickoff':decision_minutes}
    report['by_league']={}
    for league in sorted({r['league'] for r in allrows}):
      ls=[r for r in diagnostic if r['league']==league]; lp=[r for r in predictions if r['league']==league]
      report['by_league'][league]={'forecasted':len(lp),'selected_research':len(ls),'research_correct':sum(x['won'] for x in ls),'research_hit_rate':round(sum(x['won'] for x in ls)/len(ls),4) if ls else None}
    if out:
     out=Path(out);out.mkdir(parents=True,exist_ok=True)
     flat=[]
     for r in predictions:
      s={k:r[k] for k in ('league','season','date','kickoff','home','away','hg','ag','lambda_home','lambda_away','dc_market','dc_p','calibrated_market_count','xg_training_matches','timing_verified','result_timing_verified')}
      for m,p in r['probs'].items():s['p_'+m.lower().replace(' ','_').replace('.','')]=round(p,5)
      s.update({k:r['context'][k] for k in ('home_rest_days','away_rest_days','home_travel_km','away_travel_km','home_matches_last_14d','away_matches_last_14d','squad_data_available','squad_depth_available','timezone_crossing_available','motivation_data_available')})
      flat.append(s)
     write_csv(out/'predictions.csv',flat);write_csv(out/'research_top5.csv',diagnostic);write_csv(out/'priced_selections.csv',priced)
     (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
     write_dashboard(out/'dashboard.html',report,diagnostic)
    return report


def run_analyze(history,fixtures,context_store=None,odds=None,out=None,fit_days=28,warmup=35,decision_minutes=60,threshold=.80,min_ev=.04):
    """Safe future-only analysis; no current fixtures supplied -> no live predictions."""
    context_store=context_store or {};odds=odds or {};results=[];research=[];prices=[]
    for row in fixtures:
      at=row['kickoff_dt']-timedelta(minutes=decision_minutes)
      previous=[x for x in history if x['league']==row['league'] and x['result_available_dt']<at]
      model=fit_goals(previous,at,row['league'],context_store,min_history=warmup)
      if model is None:continue
      x,ctx=features(row,[r for r in history if r['kickoff_dt']<at],context_store,at)
      # Require previous-season validation for calibrators; no live use without it.
      raw=predict(model,row,x);dc='1X' if raw['probs']['1X']>=raw['probs']['X2'] else 'X2'
      entry={'date':date_key(row),'league':row['league'],'kickoff':iso(row['kickoff_dt']),'home':row['home'],'away':row['away'],'dc_market':dc,'dc_p':round(raw['probs'][dc],5),'lambda_home':raw['lambda_home'],'lambda_away':raw['lambda_away'],'xg_training_matches':model.xg_training_matches,'context_features_present':ctx,'bet_signal':'NO BET: no proven cross-season calibration/verified price'}
      results.append(entry)
    for day in sorted({r['date'] for r in results}):
      research += sorted((x for x in results if x['date']==day and x['dc_p']>=threshold),key=lambda r:-r['dc_p'])[:5]
    obj={'status':'UNPRICED_ANALYSIS_ONLY','fixtures_analysed':len(results),'research_candidates':len(research),'priced_bets':0,'warning':'Live mode is deliberately NON-TRADING until validated calibrators, licensed odds, fixture provenance, and explicit future deployment checks exist.'}
    if out:
     out=Path(out);out.mkdir(parents=True,exist_ok=True)
     write_csv(out/'analysis.csv',[{k:v for k,v in x.items() if k!='context_features_present'} for x in results]);write_csv(out/'research_top5.csv',[{k:v for k,v in x.items() if k!='context_features_present'} for x in research]);(out/'status.json').write_text(json.dumps(obj,indent=2))
    return obj


def write_dashboard(path,summary,diagnostic):
    from html import escape
    import json
    rows=[]
    for x in diagnostic:
      color='#67eab0' if x['won'] else '#ff929d'
      rows.append('<tr>'+''.join(f'<td>{escape(str(x.get(k,"")))}</td>' for k in ('date','league','home','away','market','probability','hg','ag'))+f'<td style="color:{color};font-weight:700">{"WIN" if x["won"] else "LOSS"}</td></tr>')
    d=summary['diagnostic'];p=summary['priced']; title='OMEGA V10 — Historical Research'
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>:root{{color-scheme:dark;font-family:system-ui;background:#080e18;color:#f0f5ff}}body{{max-width:1000px;margin:0 auto;padding:22px}}header{{background:#14243a;border:1px solid #274261;border-radius:22px;padding:23px}}h1{{margin:0}}p{{color:#aec0d5;line-height:1.6}}section{{background:#101b2b;border:1px solid #263a55;border-radius:18px;padding:18px;margin:16px 0}}.nums{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.metric{{background:#182a3e;padding:14px;border-radius:12px}}.metric b{{font-size:25px;display:block}}small{{color:#acc0d2}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:640px}}td,th{{padding:10px;border-bottom:1px solid #31425a;text-align:left;font-size:13px}}th{{color:#c2d5eb}}.warn{{padding:14px;background:#3a2c18;border-radius:10px;color:#ffe0a6}}@media(max-width:550px){{.nums{{grid-template-columns:1fr 1fr}}}}</style></head><body><header><small>BACKTEST · EXPLORATORY</small><h1>PRO-PITCH OMEGA V10</h1><p>Research replay. Manually transcribed examples with approximated timestamps are NOT independently verified betting backtests.</p></header><section><div class="nums"><div class="metric"><b>{d['picks']}</b><small>Research picks</small></div><div class="metric"><b>{d['wins']}</b><small>Correct</small></div><div class="metric"><b>{d['hit_rate']*100 if d['hit_rate'] is not None else 0:.1f}%</b><small>Hit rate</small></div></div></section><section class="warn">Priced selections: {p['picks']}. Win rate is not ROI. No bookmaker quotes and previous-season calibration means no validated betting advantage. This test's historical timestamps are unverified.</section><section><h2>Model diagnostics</h2><p>1X2 log loss: {summary['one_x_two_log_loss']} · Brier: {summary['one_x_two_brier']} · Mean goals forecast error: {summary['mean_goal_error_pct']}%.</p><p>Baselines on selected fixtures: Always 1X: {d.get('baseline_always_1X_on_picks')} · Always Over 1.5: {d.get('baseline_over_1_5_on_picks')}.</p></section><section><h2>Every research selection</h2><div class="scroll"><table><thead><tr><th>Date</th><th>League</th><th>Home</th><th>Away</th><th>Market</th><th>Model probability</th><th>Home goals</th><th>Away goals</th><th>Result</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section></body></html>'''
    Path(path).write_text(html,encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    subs=ap.add_subparsers(dest='command',required=True)
    for kind in ('backtest','analyze'):
      s=subs.add_parser(kind)
      s.add_argument('--matches',action='append',required=True,help='Completed match results, multiple allowed')
      if kind=='analyze':s.add_argument('--fixtures',required=True,help='Upcoming fixture CSV without results')
      s.add_argument('--context',help='As-of situational snapshots CSV')
      s.add_argument('--odds',help='Captured bookmaker quote snapshots CSV')
      s.add_argument('--out',required=True,help='Output directory')
      s.add_argument('--warmup',type=int,default=35)
      s.add_argument('--min-edge',type=float,default=.04)
      s.add_argument('--threshold',type=float,default=.80)
    a=ap.parse_args()
    hist=load_matches(a.matches);context=load_context(a.context);odds=load_odds(a.odds)
    if a.command=='backtest':result=run_backtest(hist,context,odds,out=a.out,warmup=a.warmup,min_ev=a.min_edge,threshold=a.threshold)
    else:result=run_analyze(hist,load_matches([a.fixtures],False),context,odds,out=a.out,warmup=a.warmup,min_ev=a.min_edge,threshold=a.threshold)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
