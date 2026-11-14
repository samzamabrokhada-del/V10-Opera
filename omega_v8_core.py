#!/usr/bin/env python3
"""OMEGA V8: timestamp-safe football research engine. Not a proven betting system.

Usage: python omega_v8.py --matches fixtures.csv [--odds timestamped_odds.csv] --out results/

Only records with *actual timezone-aware kickoff timestamps* are accepted. No
matchweek-time approximations or post-match odds are accepted. Predictions are
computed before score outcomes become available and match groups with identical
kickoff times update atomically. Evaluation requires 5+ seasons and forward data.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit, logsumexp, gammaln
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression

MARKETS = ('Home win', 'Draw', 'Away win', '1X', 'X2', '12',
           'Over 1.5', 'Under 2.5', 'Over 2.5', 'Under 3.5',
           'BTTS Yes', 'BTTS No')
CALIBRATE_KEYS = MARKETS
CORE_FEATURES = MARKETS


def parse_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(str(s).strip().replace('Z','+00:00'))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f'Timezone-aware timestamp required, got {s!r}')
    return dt.astimezone(timezone.utc)


def read_matches(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        required={'season','kickoff','home','away','home_goals','away_goals'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Match CSV missing columns: {sorted(required-set(reader.fieldnames or []))}')
        rows=[]
        for raw in reader:
            r={x:raw[x].strip() for x in required}
            r['kickoff_dt']=parse_dt(r['kickoff'])
            r['hg']=int(r['home_goals']);r['ag']=int(r['away_goals'])
            if r['hg']<0 or r['ag']<0:raise ValueError('No future, unplayed, or negative scores allowed')
            if not all(r[k] for k in ('season','home','away')) or r['home']==r['away']:
                raise ValueError('Invalid fixture / team')
            rows.append(r)
    rows.sort(key=lambda r:(r['kickoff_dt'],r['home'],r['away']))
    seen=set()
    for r in rows:
        key=(r['season'],r['home'],r['away'])
        if key in seen:raise ValueError(f'Duplicate season fixture {key}')
        seen.add(key)
    return rows


def read_odds(path:Path|None, cutoff_hours:float=24.) -> dict:
    """Input quotes must include an actual capture timestamp, not just kickoff."""
    quotes=defaultdict(list)
    if path is None:return quotes
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        needed={'kickoff','home','away','market','decimal_odds','recorded_at','bookmaker'}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f'Odds CSV missing {sorted(needed-set(reader.fieldnames or []))}')
        for x in reader:
            market=x['market'].strip()
            if market not in MARKETS:raise ValueError(f'Unknown market: {market}')
            key=(parse_dt(x['kickoff']),x['home'].strip(),x['away'].strip(),market)
            quote={'bookmaker':x['bookmaker'].strip(),'odds':float(x['decimal_odds']),
                   'recorded_at':parse_dt(x['recorded_at'])}
            if not math.isfinite(quote['odds']) or quote['odds']<=1:raise ValueError('Invalid odds')
            if quote['recorded_at']>=key[0]:
                raise ValueError('Odds timestamp must be strictly before kickoff (look-ahead risk)')
            if (key[0]-quote['recorded_at']).total_seconds()>cutoff_hours*3600:
                continue
            quotes[key].append(quote)
    for key in quotes:
        # Only real pre-kickoff quotes. Prefer highest offered odds (same market).
        quotes[key].sort(key=lambda q:(-q['odds'],-q['recorded_at'].timestamp()))
    return quotes


def settle(m:str,h:int,a:int)->bool:
    if h<0 or a<0:raise ValueError('Invalid goals')
    if m=='1X':return h>=a
    if m=='X2':return a>=h
    if m=='12':return h!=a
    if m=='Home win':return h>a
    if m=='Draw':return h==a
    if m=='Away win':return a>h
    if m=='Over 1.5':return h+a>=2
    if m=='Under 2.5':return h+a<=2
    if m=='Over 2.5':return h+a>=3
    if m=='Under 3.5':return h+a<=3
    if m=='BTTS Yes':return h>0 and a>0
    if m=='BTTS No':return h==0 or a==0
    raise ValueError(m)


def rho_safe(rho:float,lh:np.ndarray|float,la:np.ndarray|float):
    # Bound DC rho to keep the four corrected low-score cells positive.
    lh=np.asarray(lh);la=np.asarray(la)
    limit=np.minimum(.12, np.minimum(.90/np.maximum(lh*la,1e-8),
                       np.minimum(.90/np.maximum(lh,1e-8),.90/np.maximum(la,1e-8))))
    return rho*limit/.12


def dc_tau(h,a,lh,la,rho):
    ro=rho_safe(rho,lh,la)
    if h==0 and a==0:return 1-lh*la*ro
    if h==0 and a==1:return 1+lh*ro
    if h==1 and a==0:return 1+la*ro
    if h==1 and a==1:return 1-ro
    return np.ones_like(np.asarray(lh),dtype=float)


def score_matrix(lh:float,la:float,rho:float=0.,tail=1e-10)->np.ndarray:
    if not (0<lh<20 and 0<la<20):raise ValueError('Goal rates outside allowed range')
    n=int(max(12,poisson.ppf(1-tail/2,max(lh,la))))+1
    nums=np.arange(n)
    mat=np.outer(poisson.pmf(nums,lh),poisson.pmf(nums,la))
    for h,a in ((0,0),(0,1),(1,0),(1,1)):
        mat[h,a]*=float(dc_tau(h,a,lh,la,rho))
    if np.any(mat<0):raise ValueError('Invalid DC adjustment')
    mat/=mat.sum() # truncation error, typically under 1e-10
    return mat


def mask(m:str,n:int)->np.ndarray:
    h,a=np.indices((n,n))
    if m=='Home win':return h>a
    if m=='Draw':return h==a
    if m=='Away win':return h<a
    if m=='1X':return h>=a
    if m=='X2':return h<=a
    if m=='12':return h!=a
    if m=='Over 1.5':return h+a>=2
    if m=='Under 2.5':return h+a<=2
    if m=='Over 2.5':return h+a>=3
    if m=='Under 3.5':return h+a<=3
    if m=='BTTS Yes':return (h>0)&(a>0)
    if m=='BTTS No':return (h==0)|(a==0)
    raise ValueError(m)


def model_probs(matrix:np.ndarray)->dict[str,float]:
    assert matrix.ndim==2 and matrix.shape[0]==matrix.shape[1]
    if not np.isfinite(matrix).all() or (matrix<0).any() or abs(matrix.sum()-1)>1e-8:
        raise ValueError('Invalid scoreline matrix')
    return {m:float(matrix[mask(m,matrix.shape[0])].sum()) for m in MARKETS}


@dataclass
class GoalFit:
    intercept:float
    home_advantage:float
    rho:float
    attack:dict[str,float]
    defence:dict[str,float]
    league_mean:float
    training_matches:int

    def rates(self,home:str,away:str)->tuple[float,float]:
        ah=self.attack.get(home,0.);da=self.defence.get(away,0.)
        aa=self.attack.get(away,0.);dh=self.defence.get(home,0.)
        return (float(np.exp(np.clip(self.intercept+self.home_advantage+ah-da,-3.9,2.5))),
                float(np.exp(np.clip(self.intercept+aa-dh,-3.9,2.5))))


def fit_dixon_coles(history:list[dict],asof:datetime,half_life_days:float=180.,max_years:float=3.)->GoalFit:
    hist=[r for r in history if 0<(asof-r['kickoff_dt']).total_seconds()<=max_years*366*86400]
    if not hist:raise ValueError('No strictly earlier matches available')
    teams=sorted({r['home'] for r in hist}|{r['away'] for r in hist})
    ti={s:i for i,s in enumerate(teams)};nt=len(ti)
    hi=np.array([ti[r['home']] for r in hist]);ai=np.array([ti[r['away']] for r in hist])
    hg=np.array([r['hg'] for r in hist]);ag=np.array([r['ag'] for r in hist])
    age=np.array([(asof-r['kickoff_dt']).total_seconds()/86400 for r in hist])
    w=np.exp(-np.log(2)*age/half_life_days)
    league_mean=float(np.average(hg+ag,weights=w))
    logfac=gammaln(hg+1)+gammaln(ag+1)
    intercept=math.log(max(.35,league_mean/2))
    x0=np.zeros(3+2*nt);x0[0]=intercept;x0[1]=.15;x0[2]=-.02
    bounds=[(-1.3,1.6),(-.4,.7),(-.12,.12)]+[(-1.8,1.8)]*(2*nt)
    def loss(x):
        mu=x[0];homeadv=x[1];rho=x[2]
        attack=x[3:3+nt];defence=x[3+nt:]
        # Center the attack/defence components to identify the intercept.
        att=attack-attack.mean();df=defence-defence.mean()
        lh=np.exp(np.clip(mu+homeadv+att[hi]-df[ai],-3.9,2.5))
        la=np.exp(np.clip(mu+att[ai]-df[hi],-3.9,2.5))
        tau=np.ones(len(hist))
        for h,a in ((0,0),(0,1),(1,0),(1,1)):
            idx=(hg==h)&(ag==a)
            if np.any(idx):tau[idx]=dc_tau(h,a,lh[idx],la[idx],rho)
        logp=hg*np.log(lh)-lh+ag*np.log(la)-la-logfac+np.log(np.maximum(1e-12,tau))
        # Partial pooling helps promoted teams and sparse early-season statistics.
        penalty=1.6*(np.dot(att,att)+np.dot(df,df)) + .4*(homeadv**2) + .02*(rho/.12)**2
        return float(-(w@logp)+penalty)
    opt=minimize(loss,x0,bounds=bounds,method='L-BFGS-B',options={'maxiter':130,'ftol':1e-7})
    x=opt.x
    att=x[3:3+nt]-x[3:3+nt].mean();df=x[3+nt:]-x[3+nt:].mean()
    return GoalFit(float(x[0]),float(x[1]),float(x[2]),
                   {k:float(att[i]) for k,i in ti.items()},
                   {k:float(df[i]) for k,i in ti.items()},league_mean,len(hist))


def elo_probs(diff:float,draw_intercept:float,draw_slope:float)->np.ndarray:
    # Elo point difference -> log-odds; draw probability decreases with mismatch.
    win_cond=float(expit(diff*math.log(10)/400))
    draw=float(expit(draw_intercept-draw_slope*abs(diff)/400))
    return np.array([(1-draw)*win_cond,draw,(1-draw)*(1-win_cond)])


def replay_elo(history:list[dict],p:np.ndarray, predict_match=None):
    k,homeedge,drawint,drawslope=[float(v) for v in p]
    elo=defaultdict(lambda:1500.)
    preds=[]
    for _,group in by_kickoff(history):
        changes=defaultdict(float)
        for row in group:
            h,a=row['home'],row['away']
            pr=elo_probs(elo[h]+homeedge-elo[a],drawint,drawslope)
            preds.append((row,pr))
            actual=1. if row['hg']>row['ag'] else .5 if row['hg']==row['ag'] else 0.
            d=k*(actual-(pr[0]+.5*pr[1]))
            changes[h]+=d;changes[a]-=d
        for team,val in changes.items():elo[team]+=val
    return elo,preds


def fit_elo_params(history:list[dict],asof:datetime)->np.ndarray:
    if len(history)<130:return np.array([23.,50.,-1.0,.8]) # warm-up; not independently validated
    rec=[r for r in history if 0<(asof-r['kickoff_dt']).total_seconds()<=3*366*86400]
    if len(rec)<130:return np.array([23.,50.,-1.0,.8])
    def loss(x):
        _,preds=replay_elo(rec,x)
        val=0.
        for r,ps in preds:
            ind=0 if r['hg']>r['ag'] else 1 if r['hg']==r['ag'] else 2
            weight=.5**((asof-r['kickoff_dt']).total_seconds()/86400/365.)
            val-=weight*math.log(max(1e-12,ps[ind]))
        return val
    opt=minimize(loss,np.array([23.,50.,-1.0,.8]),method='L-BFGS-B',
                 bounds=[(2,90),(-100,180),(-2.7,-.15),(.05,5.)],options={'maxiter':65})
    return np.array(opt.x)


def by_kickoff(rows:list[dict]):
    ix=0
    while ix<len(rows):
        j=ix+1
        while j<len(rows) and rows[j]['kickoff_dt']==rows[ix]['kickoff_dt']:j+=1
        yield rows[ix]['kickoff_dt'],rows[ix:j]
        ix=j


def tilt_1x2(matrix:np.ndarray,elo:np.ndarray,weight:float)->np.ndarray:
    if weight<=0:return matrix
    p=model_probs(matrix)
    target=(1-weight)*np.array([p['Home win'],p['Draw'],p['Away win']])+weight*elo
    ans=matrix.copy();n=len(ans)
    for m,t in zip(('Home win','Draw','Away win'),target):
        sel=mask(m,n)
        ans[sel]*=float(t)/max(1e-12,p[m])
    ans/=ans.sum()
    return ans


def fit_blend_weight(previous_predictions:list[dict])->float:
    usable=[r for r in previous_predictions if r.get('elo_vector') is not None]
    if len(usable)<100:return 0.
    dp=np.array([r['dc_vector'] for r in usable]);ep=np.array([r['elo_vector'] for r in usable])
    y=np.array([r['outcome_idx'] for r in usable]);idx=np.arange(len(y))
    f=lambda w:float(-np.log(np.maximum((1-w)*dp[idx,y]+w*ep[idx,y],1e-12)).mean())
    return float(minimize_scalar(f,bounds=(0,1),method='bounded').x)


def train_calibrators(previous_predictions:list[dict],min_count=100):
    trained={}
    if len(previous_predictions)<min_count:return trained
    for market in CALIBRATE_KEYS:
        p=np.array([r['raw_probs'][market] for r in previous_predictions])
        y=np.array([int(settle(market,r['hg'],r['ag'])) for r in previous_predictions])
        if y.sum()<15 or len(y)-y.sum()<15:continue
        log_odds=logit(np.clip(p,1e-5,1-1e-5)).reshape(-1,1)
        clf=LogisticRegression(C=2.,max_iter=300).fit(log_odds,y)
        trained[market]=clf
    return trained


def recalibrate_matrix(matrix:np.ndarray, calibrators:dict)->np.ndarray:
    """Independent Platt targets are reconciled by a regularized KL projection.

    All *output* market probabilities are then recalculated from the *same*
    coherent adjusted scoreline matrix. This avoids contradictions such as
    P(1X) != P(Home)+P(Draw) introduced by separate market calibration.
    """
    if not calibrators:return matrix
    p=model_probs(matrix)
    active=[k for k in CORE_FEATURES if k in calibrators]
    if not active:return matrix
    targets=np.array([float(calibrators[k].predict_proba(
                     [[float(logit(np.clip(p[k],1e-5,1-1e-5)))]])[0,1]) for k in active])
    n=matrix.shape[0];base=np.log(np.maximum(matrix.ravel(),1e-300))
    feat=np.stack([mask(k,n).ravel().astype(float) for k in active],axis=1)
    reg=.025
    def fgrad(z):
        logits=base+feat@z
        logZ=float(logsumexp(logits))
        q=np.exp(logits-logZ)
        val=logZ-z@targets+reg/2*(z@z)
        grad=feat.T@q-targets+reg*z
        return val,grad
    fit=minimize(fgrad,np.zeros(len(active)),jac=True,method='L-BFGS-B',
                 bounds=[(-5,5)]*len(active),options={'maxiter':90})
    logits=base+feat@fit.x
    q=np.exp(logits-logsumexp(logits))
    return q.reshape(matrix.shape)


def ece(probs,ys,bins=10):
    if len(probs)==0:return None,[]
    p=np.asarray(probs);y=np.asarray(ys)
    table=[];error=0.
    for b in range(bins):
        maskbin=(p>=b/bins)&((p<(b+1)/bins) if b<bins-1 else (p<=1))
        count=int(maskbin.sum())
        if count:
            pred=float(p[maskbin].mean());observed=float(y[maskbin].mean())
            error+=count/len(p)*abs(pred-observed)
            table.append({'bin':f'{b/bins:.1f}–{(b+1)/bins:.1f}',
                          'n':count,'predicted':round(pred,4),'observed':round(observed,4)})
    return round(error,5),table


def metric_report(rows:list[dict])->dict:
    out={}
    if not rows:return out
    for market in MARKETS:
        y=np.array([int(settle(market,r['hg'],r['ag'])) for r in rows])
        p=np.array([r['probs'][market] for r in rows]);p=np.clip(p,1e-10,1-1e-10)
        cal,table=ece(p,y)
        out[market]={'n':len(rows),'observed_rate':round(float(y.mean()),5),
                     'mean_predicted':round(float(p.mean()),5),
                     'brier':round(float(np.mean((p-y)**2)),5),
                     'log_loss':round(float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),5),
                     'ece':cal,'reliability':table}
    return out


def evaluate(matches:list[dict],quotes:dict,refit_every=25,min_edge=.04,
             min_probability=.55,max_per_day=5,day_timezone="Africa/Johannesburg"):
    local_zone=ZoneInfo(day_timezone)
    if not matches:return [],{},[]
    prior=[];allrecords=[];seasons=list(dict.fromkeys(r['season'] for r in matches))
    calibrators={};blend=0.;elo_p=np.array([23.,50.,-1.0,.8]);elo=defaultdict(lambda:1500.)
    season_tag=None;model=None;fit_n=0;last_fit_time=None
    accepted_day=defaultdict(int)
    for when,group in by_kickoff(matches):
        season=group[0]['season']
        if any(r['season']!=season for r in group):
            raise ValueError('Simultaneous kickoffs across different seasons; fix season labels')
        if season!=season_tag:
            # Annual training uses PREVIOUS seasons only, not the season being evaluated.
            earlier=[r for r in allrecords if r['season']!=season]
            calibrators=train_calibrators(earlier)
            blend=fit_blend_weight(earlier)
            elo_p=fit_elo_params(prior,when)
            elo,_=replay_elo(prior,elo_p)
            elo=defaultdict(lambda:1500.,elo)
            model=None;fit_n=0;season_tag=season
        if len(prior)>=25 and (model is None or len(prior)-fit_n>=refit_every):
            model=fit_dixon_coles(prior,when)
            fit_n=len(prior);last_fit_time=when
        changes=defaultdict(float)
        for r in group:
            if model is None:
                # No trained score model. Keep these early fixtures for training,
                # but do not issue/score predictions with fabricated probabilities.
                pred=None
            else:
                lh,la=model.rates(r['home'],r['away'])
                base=score_matrix(lh,la,model.rho)
                e=elo_probs(elo[r['home']]+elo_p[1]-elo[r['away']],elo_p[2],elo_p[3])
                dc=model_probs(base)
                combined=tilt_1x2(base,e,blend)
                raw=model_probs(combined)
                final=recalibrate_matrix(combined,calibrators)
                probs=model_probs(final)
                outcome=0 if r['hg']>r['ag'] else 1 if r['hg']==r['ag'] else 2
                pred={'season':season,'kickoff':when.isoformat(),
                      'home':r['home'],'away':r['away'],'hg':r['hg'],'ag':r['ag'],
                      'model_train_n':fit_n,'calibrated':bool(calibrators),
                      'blend_weight':blend,'expected_goals':lh+la,
                      'expected_home_goals':lh,'expected_away_goals':la,
                      'raw_probs':raw,'probs':probs,'outcome_idx':outcome,
                      'dc_vector':[dc['Home win'],dc['Draw'],dc['Away win']],
                      'elo_vector':list(map(float,e)),
                      'market':None,'odds':None,'edge':None,'bet_return':None,
                      'bookmaker':None,'bet_result':None}
                # Do not choose markets from raw probability alone: require real odds.
                candidates=[]
                for m in MARKETS:
                    key=(when,r['home'],r['away'],m)
                    if key not in quotes:continue
                    q=quotes[key][0]
                    edge=probs[m]*q['odds']-1.
                    if edge>=min_edge and probs[m]>=min_probability:
                        candidates.append((edge,m,q))
                day=when.astimezone(local_zone).date().isoformat()
                candidates=[c for c in candidates if c[1] in calibrators]
                if candidates and accepted_day[day]<max_per_day and len(calibrators)>=3:
                    edge,m,q=max(candidates,key=lambda c:c[0])
                    pred.update(market=m,odds=q['odds'],edge=edge,bookmaker=q['bookmaker'],
                                bet_result='WIN' if settle(m,r['hg'],r['ag']) else 'LOSS',
                                bet_return=(q['odds']-1) if settle(m,r['hg'],r['ag']) else -1.)
                    accepted_day[day]+=1
                allrecords.append(pred)
            # Grouped same-time updates: forecasts above see only history before kickoff.
            ep=elo_probs(elo[r['home']]+elo_p[1]-elo[r['away']],elo_p[2],elo_p[3])
            actual=1. if r['hg']>r['ag'] else .5 if r['hg']==r['ag'] else 0.
            d=elo_p[0]*(actual-(ep[0]+.5*ep[1]))
            changes[r['home']]+=d;changes[r['away']]-=d
        prior.extend(group)
        for team,v in changes.items():elo[team]+=v
    # Report per-season statistics. Training first 25+ matches have no forecasts.
    reports={}
    for season in seasons:
        pred=[r for r in allrecords if r['season']==season]
        selections=[r for r in pred if r['market'] is not None]
        actual_goals=sum(r['hg']+r['ag'] for r in pred)
        predicted_goals=sum(r['expected_goals'] for r in pred)
        goal_diff=((predicted_goals/actual_goals)-1)*100 if actual_goals else None
        profit=sum(r['bet_return'] for r in selections)
        # Baseline uses actual Over 1.5 quotes on the SAME full set of forecast fixtures.
        baseline_returns=[]
        for r in pred:
            k=(parse_dt(r['kickoff']),r['home'],r['away'],'Over 1.5')
            if k in quotes:
                price=quotes[k][0]['odds']
                baseline_returns.append(price-1 if settle('Over 1.5',r['hg'],r['ag']) else -1.)
        oy=np.array([0 if r['hg']<r['ag'] else 1 if r['hg']==r['ag'] else 2 for r in pred])
        mat3=np.array([[r['probs'][k] for k in ('Home win','Draw','Away win')] for r in pred])
        multiclass_brier=float(np.mean(np.sum((mat3-np.eye(3)[oy])**2,axis=1))) if pred else None
        multiclass_logloss=float(np.mean(-np.log(np.maximum(1e-12,mat3[np.arange(len(pred)),oy])))) if pred else None
        reports[season]={
            'predictions':len(pred),'picks_with_real_odds':len(selections),
            'wins':sum(r['bet_result']=='WIN' for r in selections),
            'win_rate':round(sum(r['bet_result']=='WIN' for r in selections)/len(selections),4) if selections else None,
            'yield_per_unit':round(profit/len(selections),5) if selections else None,
            'mean_predicted_total_goals':round(predicted_goals/len(pred),4) if pred else None,
            'mean_actual_total_goals':round(actual_goals/len(pred),4) if pred else None,
            'mean_goal_error_pct':round(goal_diff,2) if goal_diff is not None else None,
            'meets_2pct_goal_scale_target':abs(goal_diff)<=2 if goal_diff is not None else None,
            'always_over_1_5_hit_rate':round(sum(settle('Over 1.5',r['hg'],r['ag']) for r in pred)/len(pred),4) if pred else None,
            'always_over_1_5_quoted_n':len(baseline_returns),
            'always_over_1_5_quoted_yield':round(sum(baseline_returns)/len(baseline_returns),5) if baseline_returns else None,
            'multiclass_1x2_brier':round(multiclass_brier,5) if multiclass_brier is not None else None,
            'multiclass_1x2_logloss':round(multiclass_logloss,5) if multiclass_logloss is not None else None,
            'market_metrics':metric_report(pred),
            'precalibration_market_metrics':metric_report([{**r,'probs':r['raw_probs']} for r in pred])}
    seasons_full={s:sum(r['season']==s for r in matches)>=300 for s in seasons}
    status={'seasons_in_input':seasons,'number_of_seasons':len(seasons),
            'full_seasons_300plus_matches':seasons_full,
            'five_season_validation_possible':sum(seasons_full.values())>=5,
            'real_odds_input_supplied':bool(quotes),
            'priced_selections':sum(r['market'] is not None for r in allrecords),
            'profitability_proven':False,
            'warnings':['All included seasons have been explored previously unless explicitly withheld. No pristine forward test has been recorded.',
                        'A five-season walk-forward result requires five or more near-complete seasons (300+ matches each) with genuine kickoff timestamps.',
                        'Goal-scale target is audited out-of-sample; not forcibly fitted to future goals.',
                        'No picks are made without verifiable, time-stamped pre-kickoff bookmaker odds and earlier-season market Platt calibration.']}
    return allrecords,reports,status


def save_results(pred,report,status,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'predictions.csv').open('w',newline='',encoding='utf8') as f:
        cols=['season','kickoff','home','away','hg','ag','model_train_n','calibrated','blend_weight',
              'expected_goals','expected_home_goals','expected_away_goals','market','odds','edge',
              'bet_result','bet_return','bookmaker','raw_probs','probs']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in pred:
            w.writerow({c:json.dumps(r[c]) if c in ('raw_probs','probs') else r.get(c) for c in cols})
    (out/'report.json').write_text(json.dumps({'status':status,'seasons':report},indent=2),encoding='utf8')
    with (out/'selections.csv').open('w',newline='',encoding='utf8') as f:
        cols=['season','kickoff','home','away','market','odds','edge','bookmaker','bet_result','bet_return']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({c:r.get(c) for c in cols} for r in pred if r['market'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matches',type=Path,required=True,help='CSV with real timezone-aware kickoff timestamps')
    ap.add_argument('--odds',type=Path,default=None,help='Historical sportsbook odds with recorded_at timestamps')
    ap.add_argument('--out',type=Path,default=Path('omega_v8_results'))
    ap.add_argument('--refit-every',type=int,default=25)
    ap.add_argument('--minimum-edge',type=float,default=.04)
    ap.add_argument('--minimum-probability',type=float,default=.55)
    ap.add_argument('--max-daily-picks',type=int,default=5)
    ap.add_argument('--day-timezone',default='Africa/Johannesburg',help='Timezone for daily selection limit')
    args=ap.parse_args()
    if args.refit_every<1 or args.max_daily_picks<1:ap.error('Refit and pick caps must be positive')
    if not 0<=args.minimum_edge<=1 or not 0<=args.minimum_probability<=1:ap.error('Thresholds must be in [0,1]')
    matches=read_matches(args.matches);odds=read_odds(args.odds)
    pred,report,status=evaluate(matches,odds,args.refit_every,args.minimum_edge,
                                args.minimum_probability,args.max_daily_picks,args.day_timezone)
    save_results(pred,report,status,args.out)
    print(json.dumps({'status':status,'season_summary':{k:{x:v[x] for x in ('predictions','picks_with_real_odds','win_rate','yield_per_unit','mean_goal_error_pct','always_over_1_5_hit_rate')} for k,v in report.items()},'output':str(args.out)},indent=2))

if __name__=='__main__':main()
