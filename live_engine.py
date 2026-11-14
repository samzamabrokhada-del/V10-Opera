"""V10 model bridge for genuine provider fixtures. Experimental probabilities only.

No final result is ever sent into a model predicting that fixture. Existing
forecasts are immutable after they're first recorded before kickoff.
"""
from __future__ import annotations
import copy, math
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict
import numpy as np
from omega_v10 import fit_goals, predict, features

UTC=timezone.utc
SAST=ZoneInfo('Africa/Johannesburg')
DC_THRESHOLD=.80
MAX_DAY_PICKS=5

def parse_iso(s):
    d=datetime.fromisoformat(str(s).replace('Z','+00:00'))
    if d.tzinfo is None: raise ValueError('Naive date not allowed')
    return d.astimezone(UTC)

def utcnow():return datetime.now(UTC)

def normalize_football_data(code, payload):
    out=[]
    for m in payload.get('matches',[]):
        try:
            home=m['homeTeam']['name'];away=m['awayTeam']['name']
            kickoff=parse_iso(m['utcDate'])
            if not home or not away or home==away:continue
            st=m.get('status','SCHEDULED');sc=m.get('score') or {};ft=sc.get('fullTime') or {}
            hg,ag=ft.get('home'),ft.get('away')
            is_final=st=='FINISHED' and isinstance(hg,int) and isinstance(ag,int)
            out.append({'id':'fd:'+str(m['id']),'league':code,'league_name':(m.get('competition') or {}).get('name') or code,
                        'season':str((m.get('season') or {}).get('startDate') or kickoff.date().isoformat())[:4],
                        'kickoff':kickoff.isoformat(),'home':home,'away':away,'status':'FINISHED' if is_final else st,
                        'hg':hg if is_final else None,'ag':ag if is_final else None,'source':'football-data.org',
                        'timing_verified':True})
        except (KeyError,ValueError,TypeError):continue
    return out

def normalize_api_sports(payload):
    out=[]
    for m in payload.get('response',[]):
        try:
            fix=m['fixture'];ko=parse_iso(fix['date']);status=(fix['status'] or {}).get('short','NS')
            home=m['teams']['home']['name'];away=m['teams']['away']['name']
            leag=m.get('league') or {}; goals=m.get('goals') or {}
            hg,ag=goals.get('home'),goals.get('away')
            final=status in ('FT','AET','PEN') and isinstance(hg,int) and isinstance(ag,int)
            out.append({'id':'as:'+str(fix['id']),'league':'AS'+str(leag.get('id','')),
                        'league_name':leag.get('name') or 'International / other','season':str(leag.get('season') or ko.year),
                        'kickoff':ko.isoformat(),'home':home,'away':away,'status':'FINISHED' if final else status,
                        'hg':hg if final else None,'ag':ag if final else None,'source':'API-Football',
                        'timing_verified':True})
        except (KeyError,ValueError,TypeError):continue
    return out

def _row(m):
    ko=parse_iso(m['kickoff'])
    # Score of finished matches is deliberately not made available until >3h after kickoff.
    return {'league':m['league'],'season':m['season'],'home':m['home'],'away':m['away'],
            'kickoff_dt':ko,'result_available_dt':ko+timedelta(hours=3),
            'hg':m['hg'] if m['status']=='FINISHED' else None,
            'ag':m['ag'] if m['status']=='FINISHED' else None,
            'home_xg':None,'away_xg':None,'home_npxg':None,'away_npxg':None,
            'venue_lat':None,'venue_lon':None}

def rank_select(forecasts,previous=None,threshold=DC_THRESHOLD,max_day=MAX_DAY_PICKS,now=None):
    """Preserve already locked selections; never more than max_day per SAST date."""
    previous=previous or {}
    now=now or utcnow()
    selected={str(k):v for k,v in previous.items() if any(x['id']==k for x in forecasts)}
    grouped=defaultdict(list)
    for f in forecasts:
        ko=parse_iso(f['kickoff']);date=ko.astimezone(SAST).date().isoformat()
        if float(f['probability'])>=threshold and parse_iso(f['kickoff'])>now:grouped[date].append(f)
    for d,items in grouped.items():
        items.sort(key=lambda x:(-x['probability'],x['kickoff'],x['id']))
        used=sum(1 for v in selected.values() if v['date']==d)
        for f in items:
            if used>=max_day:break
            if f['id'] in selected:continue
            selected[f['id']]={'date':d,'locked_at':f['forecast_at'],'market':f['market'],'probability':f['probability']}
            used+=1
    return selected

def analyze(matches, prior_forecasts=None,prior_selected=None, now=None):
    """Generate pre-match predictions only; never look at future results.

    `matches` are normalized provider objects. `prior_forecasts` is immutable and
    persisted between syncs to prevent retroactive revision of predictions.
    """
    now=now or utcnow();prior_forecasts=prior_forecasts or {};prior_selected=prior_selected or {}
    models={};history=defaultdict(list); upcoming=[]
    for m in matches:
        try:r=_row(m)
        except (ValueError,KeyError,TypeError):continue
        ko=r['kickoff_dt']
        if m['status']=='FINISHED' and ko+timedelta(hours=3)<now:history[m['league']].append(r)
        elif now<ko<=now+timedelta(days=9) and m['status'] not in ('POSTPONED','SUSPENDED','CANCELLED','PST','CANC'):
            upcoming.append((m,r))
    for league,items in history.items():
        try:
            # One fit per league, using only completed fixtures as of the snapshot time.
            model=fit_goals(items,now,league,min_history=30)
            if model:models[league]=model
        except Exception: pass
    forecasts=dict(prior_forecasts)
    skipped=[]
    for m,r in sorted(upcoming,key=lambda z:z[1]['kickoff_dt']):
        if m['id'] in forecasts:continue
        model=models.get(m['league'])
        if not model:
            skipped.append({'id':m['id'],'reason':'Not enough completed results to fit this competition'})
            continue
        try:
            prior=history[m['league']]
            z,known=features(r,prior,{},now)
            result=predict(model,r,z,calibrators=None)
            p=result['probs'];market='1X' if p['1X']>=p['X2'] else 'X2';chance=float(p[market])
            forecasts[m['id']]={
                'id':m['id'],'league':m['league_name'],'league_code':m['league'],
                'home':m['home'],'away':m['away'],'kickoff':m['kickoff'],
                'date':r['kickoff_dt'].astimezone(SAST).date().isoformat(),
                'market':market,'probability':round(chance,5),
                'p_home':round(float(p['Home win']),5),'p_draw':round(float(p['Draw']),5),
                'p_away':round(float(p['Away win']),5),
                'p_over15':round(float(p['Over 1.5']),5),
                'p_under35':round(float(p['Under 3.5']),5),
                'p_btts':round(float(p['BTTS Yes']),5),
                'expected_home_goals':result['lambda_home'],'expected_away_goals':result['lambda_away'],
                'forecast_at':now.isoformat(),'trained_matches':model.trained_matches,
                'calibrated':False,'context_availability':known,
                'qualification':'RESEARCH ONLY — no independent calibration or verified betting edge'}
        except Exception as ex:
            skipped.append({'id':m['id'],'reason':'Prediction unavailable: '+str(ex)[:140]})
    current=[v for k,v in forecasts.items() if any(m['id']==k and parse_iso(m['kickoff'])>now for m in matches)]
    selected=rank_select(list(forecasts.values()),prior_selected,now=now)
    finals=[]
    for m in matches:
        if m['id'] in forecasts and m['status']=='FINISHED' and m['hg'] is not None:
            f=forecasts[m['id']]
            won=(m['hg']>=m['ag']) if f['market']=='1X' else (m['ag']>=m['hg'])
            finals.append({'id':m['id'],'home':m['home'],'away':m['away'],'league':m['league_name'],'market':f['market'],
                           'probability':f['probability'],'home_goals':m['hg'],'away_goals':m['ag'],
                           'won':bool(won),'kickoff':m['kickoff'],'selected':m['id'] in selected})
    return {'models':{l:{'trained_matches':x.trained_matches,'asof':x.asof} for l,x in models.items()},
            'forecasts':forecasts,'selected':selected,'upcoming':sorted(current,key=lambda x:x['kickoff']),
            'results':sorted(finals,key=lambda x:x['kickoff'],reverse=True),'skipped':skipped}
