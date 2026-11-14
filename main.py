"""PRO-PITCH OMEGA V10 - deployable multi-league data and research server.

Requires env FOOTBALL_DATA_API_KEY or API_SPORTS_KEY; credentials never sent to clients.
Research-only predictions. No bookmaker account, bet placement or real-money action.
"""
from __future__ import annotations
import json, os, threading, time, tempfile, traceback, hmac, hashlib, html
from urllib.parse import parse_qs
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from live_engine import normalize_football_data, normalize_api_sports, analyze, utcnow, parse_iso

ROOT=Path(__file__).resolve().parent.parent
STATIC=ROOT/'site'
DATA_PATH=Path(os.environ.get('OMEGA_DATA_PATH',str(ROOT/'state'/'snapshot.json')))
FD_KEY=os.environ.get('FOOTBALL_DATA_API_KEY','').strip()
AS_KEY=os.environ.get('API_SPORTS_KEY','').strip()
ODDS_KEY=os.environ.get('ODDS_API_KEY','').strip()
LEAGUES=[x.strip().upper() for x in os.environ.get('FOOTBALL_COMPETITIONS','PL,PD,BL1,SA,FL1,DED,PPL,ELC,CL').split(',') if x.strip()]
API_SPORTS_LEAGUES=[x.strip() for x in os.environ.get('API_SPORTS_LEAGUES','').split(',') if x.strip().isdigit()]
REFRESH_MINS=max(30,int(os.environ.get('OMEGA_REFRESH_MINUTES','180')))
REQUEST_SPACING=max(0.,float(os.environ.get('FOOTBALL_API_SPACING_SECONDS','6.3')))
ODDS_ENABLED=os.environ.get('ENABLE_ODDS','false').lower()=='true'
ADMIN_TOKEN=os.environ.get('OMEGA_ADMIN_TOKEN','').strip()
REQUIRE_LOGIN=os.environ.get('OMEGA_REQUIRE_LOGIN','false').lower()=='true'
APP_PASSWORD=os.environ.get('OMEGA_APP_PASSWORD','')
SESSION_SECRET=os.environ.get('OMEGA_SESSION_SECRET','')
VAPID_PUBLIC=os.environ.get('VAPID_PUBLIC_KEY','')
VAPID_PRIVATE=os.environ.get('VAPID_PRIVATE_KEY','')
VAPID_SUBJECT=os.environ.get('VAPID_SUBJECT','mailto:omega@example.invalid')
ALLOWED_ORIGINS=[x.strip() for x in os.environ.get('OMEGA_ALLOWED_ORIGINS','').split(',') if x.strip()]
SYNC_LOCK=threading.Lock()
STATE_LOCK=threading.RLock()
STOP=threading.Event()
REFRESHING=False
LATEST_ERROR=None
LAST_STARTED=0.

app=FastAPI(title='PRO-PITCH OMEGA V10 Live Research API',version='10.1.0')
app.add_middleware(CORSMiddleware,allow_origins=ALLOWED_ORIGINS,allow_credentials=False,allow_methods=['GET','POST'],allow_headers=['Content-Type','X-Admin-Token'])

# Private single-owner login. The cookie holds no password and is HttpOnly.
# This gate covers the PWA files and every research endpoint, not just the home page.
AUTH_FAILS={}
AUTH_LOCK=threading.Lock()
COOKIE='omega_session'

def _session_token(expiry):
    message=f'omega-v10|{expiry}|{hashlib.sha256(APP_PASSWORD.encode()).hexdigest()}'.encode()
    return hmac.new(SESSION_SECRET.encode(),message,hashlib.sha256).hexdigest()

def _authorized(request:Request):
    if not REQUIRE_LOGIN:return True
    if not APP_PASSWORD or not SESSION_SECRET:return False
    val=request.cookies.get(COOKIE,'')
    try:
        exp_str,mac=val.split('.',1)
        exp=int(exp_str)
        return exp>int(time.time()) and hmac.compare_digest(mac,_session_token(exp))
    except (ValueError,TypeError):return False

@app.middleware('http')
async def private_gate(request:Request,call_next):
    path=request.url.path
    if REQUIRE_LOGIN and path not in ('/api/health','/login') and not _authorized(request):
        if path.startswith('/api/'):
            return JSONResponse({'detail':'Private OMEGA login required'},status_code=401,
                                headers={'Cache-Control':'no-store'})
        return RedirectResponse('/login',status_code=303,headers={'Cache-Control':'no-store'})
    response=await call_next(request)
    response.headers['X-Robots-Tag']='noindex, nofollow, noarchive'
    if path in ('/login','/') or path.startswith('/api/'):
        response.headers['Cache-Control']='no-store'
    return response

LOGIN_HTML="""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#081521"><title>OMEGA · Private Access</title><style>*{box-sizing:border-box}body{margin:0;background:#07121e;color:#f5f8fd;font:16px system-ui;min-height:100dvh;display:grid;place-items:center;padding:20px}.panel{width:min(100%,390px);background:#122334;border:1px solid #2b455b;padding:28px;border-radius:22px;box-shadow:0 18px 50px #0005}h1{font-size:26px;margin:8px 0}p{color:#b1c4d3;line-height:1.5}.logo{color:#6de0bd;font-weight:800;letter-spacing:.12em}label{display:block;font-size:13px;margin:20px 0 7px}input{width:100%;border:1px solid #4a6479;border-radius:11px;padding:14px;font-size:16px;background:#091a29;color:white}button{width:100%;border:0;border-radius:11px;padding:14px;font-size:16px;font-weight:750;background:#69dfba;color:#06251e;margin-top:16px}.err{color:#ffb9b2}</style></head><body><main class="panel"><div class="logo">Ω PRO-PITCH</div><h1>OMEGA V10</h1><p>Private football intelligence. Enter your personal access password.</p><form action="/login" method="post"><label for="pw">Password</label><input id="pw" type="password" name="password" required autocomplete="current-password" autofocus><button type="submit">Unlock OMEGA</button></form>__ERROR__<p style="font-size:12px">Research only · Personal access</p></main></body></html>"""

@app.get('/login',include_in_schema=False)
def login_screen(request:Request):
    if _authorized(request):return RedirectResponse('/',status_code=303)
    return HTMLResponse(LOGIN_HTML.replace('__ERROR__',''),headers={'Cache-Control':'no-store'})

@app.post('/login',include_in_schema=False)
async def login_submit(request:Request):
    if not REQUIRE_LOGIN:return RedirectResponse('/',status_code=303)
    ip=request.client.host if request.client else 'unknown'
    now=time.monotonic()
    with AUTH_LOCK:
        n,until=AUTH_FAILS.get(ip,(0,0.))
        if until and now>until:n=0
        if n>=7 and now<until:
            return HTMLResponse(LOGIN_HTML.replace('__ERROR__','<p class="err">Too many attempts. Try again later.</p>'),status_code=429,headers={'Retry-After':'900','Cache-Control':'no-store'})
    raw=await request.body()
    password=parse_qs(raw[:2048].decode('utf8',errors='replace')).get('password',[''])[0]
    if not APP_PASSWORD or not SESSION_SECRET or not hmac.compare_digest(password,APP_PASSWORD):
        with AUTH_LOCK:AUTH_FAILS[ip]=(n+1,now+900)
        return HTMLResponse(LOGIN_HTML.replace('__ERROR__','<p class="err">Incorrect password.</p>'),status_code=401,headers={'Cache-Control':'no-store'})
    with AUTH_LOCK:AUTH_FAILS.pop(ip,None)
    expiry=int(time.time())+30*86400
    response=RedirectResponse('/',status_code=303)
    secure=bool(os.environ.get('RENDER')) or request.url.scheme=='https'
    response.set_cookie(COOKIE,f'{expiry}.{_session_token(expiry)}',max_age=30*86400,
                        secure=secure,httponly=True,samesite='strict',path='/')
    response.headers['Cache-Control']='no-store'
    return response

@app.post('/logout',include_in_schema=False)
def logout():
    response=RedirectResponse('/login',status_code=303)
    response.delete_cookie(COOKIE,path='/')
    response.headers['Cache-Control']='no-store'
    return response


def default_state():
 return {'version':'10.1','updated_at':None,'fixture_count':0,'matches':[],
         'forecasts':{},'selected':{},'results':[],'models':{},'upcoming':[],
         'provider_errors':[],'subscribers':[],'odds':{},'source':'No data provider connected'}

def read_state():
 with STATE_LOCK:
  if not DATA_PATH.exists():return default_state()
  try:return {**default_state(),**json.loads(DATA_PATH.read_text(encoding='utf8'))}
  except (ValueError,OSError):return default_state()

def write_state(state):
 with STATE_LOCK:
  DATA_PATH.parent.mkdir(parents=True,exist_ok=True)
  # Atomic replacement prevents clients from reading partially-written JSON.
  with tempfile.NamedTemporaryFile(mode='w',encoding='utf8',dir=DATA_PATH.parent,delete=False) as f:
   json.dump(state,f,ensure_ascii=False,separators=(',',':'),default=str)
   tmp=f.name
  os.replace(tmp,DATA_PATH)

def fetch_fd(client,code):
 res=client.get(f'https://api.football-data.org/v4/competitions/{code}/matches',
                headers={'X-Auth-Token':FD_KEY},timeout=18)
 res.raise_for_status()
 return normalize_football_data(code,res.json())

def fetch_as(client,code):
 # League IDs come from the user's own API-Sports dashboard. Unknown IDs are never guessed.
 year=int(datetime.now(timezone.utc).year)
 # National-team competitions commonly use calendar years; users may override globally.
 if os.environ.get('API_SPORTS_SEASON'):
  year=int(os.environ['API_SPORTS_SEASON'])
 else:
  year=year-1 if datetime.now(timezone.utc).month<7 else year
 res=client.get('https://v3.football.api-sports.io/fixtures',params={'league':code,'season':year},
                headers={'x-apisports-key':AS_KEY},timeout=18)
 res.raise_for_status()
 p=res.json()
 if p.get('errors'):raise ValueError('Provider error: '+str(p['errors'])[:180])
 return normalize_api_sports(p)

def _normalise_name(s):
 import re
 return re.sub(r'\b(fc|cf|afc|sc|club|united|city)\b','',re.sub('[^a-z0-9 ]',' ',s.lower())).replace('  ',' ').strip()

def _match_odds(name,expected):
 import difflib
 a,b=_normalise_name(name),_normalise_name(expected)
 return a==b or (len(a)>5 and len(b)>5 and difflib.SequenceMatcher(None,a,b).ratio()>=.88)

def load_odds(client,upcoming):
 if not (ODDS_ENABLED and ODDS_KEY):return {}
 sports={'PL':'soccer_epl','PD':'soccer_spain_la_liga','BL1':'soccer_germany_bundesliga',
         'SA':'soccer_italy_serie_a','FL1':'soccer_france_ligue_one',
         'DED':'soccer_netherlands_eredivisie','PPL':'soccer_portugal_primeira_liga',
         'CL':'soccer_uefa_champs_league'}
 allowed={x.strip() for x in os.environ.get('ODDS_COMPETITIONS','PL,PD').split(',')}
 out={}
 for code in set(x['league_code'] for x in upcoming)&allowed:
  if code not in sports:continue
  try:
   r=client.get(f'https://api.the-odds-api.com/v4/sports/{sports[code]}/odds/',
      params={'apiKey':ODDS_KEY,'regions':os.environ.get('ODDS_REGIONS','eu'),
              'markets':'h2h','oddsFormat':'decimal'},timeout=22)
   r.raise_for_status()
   for fixture in [f for f in upcoming if f['league_code']==code]:
    if parse_iso(fixture['kickoff'])<=utcnow():continue
    found=[]
    for event in r.json():
     if abs((parse_iso(event['commence_time'])-parse_iso(fixture['kickoff'])).total_seconds())>10800:continue
     if not (_match_odds(event['home_team'],fixture['home']) and _match_odds(event['away_team'],fixture['away'])):continue
     for book in event.get('bookmakers',[]):
      for mk in book.get('markets',[]):
       if mk.get('key')!='h2h':continue
       quotes=[]
       for o in mk.get('outcomes',[]):
        market='Draw' if o['name']=='Draw' else 'Home win' if _match_odds(o['name'],fixture['home']) else 'Away win' if _match_odds(o['name'],fixture['away']) else None
        try:price=float(o['price'])
        except (TypeError,ValueError):continue
        if market and price>1:quotes.append({'market':market,'decimal_odds':price})
       if quotes: found.append({'bookmaker':book.get('title',book.get('key','Unknown')),
          'captured_at':mk.get('last_update'), 'quotes':quotes,'source':'The Odds API'})
    if found:out[fixture['id']]=found[:15]
  except Exception:
   # Odds feed is optional. A failure may never manufacture odds or a BET flag.
   continue
 return out

def _public_data(st):
 matches=st.get('matches',[])
 now=utcnow()
 active=[m for m in matches if abs((parse_iso(m['kickoff'])-now).total_seconds())<10*86400]
 active.sort(key=lambda x:x['kickoff'])
 results=st.get('results',[])
 qualified=[]
 for fid,sel in st.get('selected',{}).items():
  f=st.get('forecasts',{}).get(fid)
  if f and parse_iso(f['kickoff'])>now:
   qualified.append({**f,'locked':sel})
 qualified.sort(key=lambda x:x['kickoff'])
 return {'version':st['version'],'connected':bool(FD_KEY or AS_KEY) and st.get('updated_at') is not None,
         'configured':{'football_data':bool(FD_KEY),'api_sports':bool(AS_KEY),'odds':bool(ODDS_KEY) and ODDS_ENABLED,
                        'web_push':bool(VAPID_PUBLIC and VAPID_PRIVATE)},
         'data_provider':st.get('source'),'updated_at':st.get('updated_at'),
         'refreshing':REFRESHING,'last_error':LATEST_ERROR,
         'leagues':list(st.get('models',{}).keys()),'training':st.get('models',{}),
         'fixture_count':st.get('fixture_count',0),'recent_fixtures':active[:220],
         'upcoming':st.get('upcoming',[])[:180],
         'picks':qualified,'results':results[:60],'raw_odds':{k:v for k,v in st.get('odds',{}).items() if k in {f['id'] for f in qualified}},
         'research_only':True,'actual_betting_signals':0,
         'data_warning':'Prices and scores may be delayed; calibrated edge not established. NO BET mode.',
         'provider_errors':st.get('provider_errors',[])[:12]}

def _notify_new_picks(prev,st):
 new=set(st.get('selected',{}))-set(prev.get('selected',{}))
 if not new or not (VAPID_PUBLIC and VAPID_PRIVATE):return
 try:
  from pywebpush import webpush
 except ImportError:return
 subs=st.get('subscribers',[])
 for fid in new:
  f=st['forecasts'].get(fid)
  if not f:continue
  payload=json.dumps({'title':'OMEGA paper candidate','body':f["home"]+' vs '+f['away']+' · '+f['market']+' · uncalibrated',
         'url':'/?view=live'})
  for sub in subs:
   try:
    webpush(subscription_info=sub,data=payload,vapid_private_key=VAPID_PRIVATE,
            vapid_claims={'sub':VAPID_SUBJECT},ttl=3600)
   except Exception:pass

def sync_once(client_factory=httpx.Client):
 global LATEST_ERROR
 old=read_state(); data=[];errors=[]
 try:
  with client_factory() as client:
   if FD_KEY:
    for idx,code in enumerate(LEAGUES):
     if idx and REQUEST_SPACING:time.sleep(REQUEST_SPACING)
     try:data.extend(fetch_fd(client,code))
     except Exception as e:errors.append(f'{code}: {type(e).__name__} - {str(e)[:160]}')
   if AS_KEY:
    for idx,code in enumerate(API_SPORTS_LEAGUES):
     try:data.extend(fetch_as(client,code))
     except Exception as e:errors.append(f'API-Sports {code}: {type(e).__name__} - {str(e)[:160]}')
   if not data:
    LATEST_ERROR='; '.join(errors) or 'No fixtures available: configure a provider key.'
    updated={**old,'provider_errors':errors}
    write_state(updated)
    return updated
   ids={}
   for m in data:ids[m['id']]=m
   matches=sorted(ids.values(),key=lambda x:x['kickoff'])
   analytics=analyze(matches,old.get('forecasts'),old.get('selected'))
   odds=load_odds(client,analytics['upcoming'])
   new={**old,'updated_at':utcnow().isoformat(),'matches':matches,
       'fixture_count':len(matches),'provider_errors':errors,'source':', '.join(x for x,v in [('football-data.org',FD_KEY),('API-Football',AS_KEY)] if v),
       **{k:analytics[k] for k in ('forecasts','selected','results','models','upcoming')},
       'odds':{**old.get('odds',{}),**odds}}
   write_state(new)
   _notify_new_picks(old,new)
   LATEST_ERROR='; '.join(errors) if errors else None
   return new
 except Exception as e:
  LATEST_ERROR=f'Refresh failed: {type(e).__name__}: {str(e)[:180]}'
  return old

def _worker():
 global REFRESHING
 try:sync_once()
 finally:
  REFRESHING=False
  SYNC_LOCK.release()

def start_refresh(force=False):
 global REFRESHING,LAST_STARTED
 if not(FD_KEY or AS_KEY):return False
 if not SYNC_LOCK.acquire(blocking=False):return False
 if not force and (time.monotonic()-LAST_STARTED)<1800 and LAST_STARTED>0:
  SYNC_LOCK.release();return False
 LAST_STARTED=time.monotonic();REFRESHING=True
 threading.Thread(target=_worker,daemon=True).start()
 return True

def periodic():
 if FD_KEY or AS_KEY:start_refresh()
 while not STOP.wait(60):
  st=read_state();at=st.get('updated_at')
  if not at or (utcnow()-parse_iso(at)).total_seconds()>REFRESH_MINS*60:
   start_refresh()

@app.on_event('startup')
def on_startup():
 if REQUIRE_LOGIN and (len(APP_PASSWORD)<12 or len(SESSION_SECRET)<32):
  raise RuntimeError('OMEGA private login must be configured with a 12+ character password and 32+ character session secret')
 STOP.clear()
 if os.environ.get('OMEGA_DISABLE_SCHEDULER','false').lower()!='true':
  threading.Thread(target=periodic,daemon=True).start()

@app.on_event('shutdown')
def on_shutdown():STOP.set()

@app.get('/api/health')
def health():return {'status':'ok','app':'OMEGA V10 Live Research','time':utcnow().isoformat()}

@app.get('/api/snapshot')
def snapshot():return _public_data(read_state())

@app.post('/api/refresh')
def refresh(request:Request):
 if ADMIN_TOKEN and request.headers.get('X-Admin-Token')!=ADMIN_TOKEN:raise HTTPException(status_code=403,detail='Admin token required to refresh')
 started=start_refresh()
 return {'started':started,'reason':'started' if started else ('no data-provider credentials' if not(FD_KEY or AS_KEY) else 'running or rate-limited')}

@app.get('/api/push/public-key')
def push_key():return {'enabled':bool(VAPID_PUBLIC and VAPID_PRIVATE),'public_key':VAPID_PUBLIC if VAPID_PRIVATE else ''}

@app.post('/api/push/subscribe')
async def push_subscribe(request:Request):
 if not (VAPID_PUBLIC and VAPID_PRIVATE):raise HTTPException(status_code=503,detail='Web push is not configured')
 sub=await request.json()
 if not isinstance(sub,dict) or not str(sub.get('endpoint','')).startswith('https://') or not all(x in sub.get('keys',{}) for x in ('auth','p256dh')):raise HTTPException(422,'Invalid push subscription')
 st=read_state();items=st.get('subscribers',[])
 if not any(x.get('endpoint')==sub['endpoint'] for x in items):items.append({'endpoint':sub['endpoint'],'keys':sub['keys']})
 st['subscribers']=items[-250:];write_state(st)
 return {'subscribed':True}

app.mount('/icons',StaticFiles(directory=STATIC/'icons'),name='icons')
def static_route_for(filename):
 async def serve_bundled_file():return FileResponse(STATIC/filename)
 serve_bundled_file.__name__='serve_'+filename.replace('.','_').replace('-','_')
 return serve_bundled_file

for fname in ('app.js','live-ui.js','data.js','style.css','sw.js','manifest.webmanifest'):
 app.add_api_route('/'+fname,static_route_for(fname),methods=['GET'],include_in_schema=False)

@app.get('/',include_in_schema=False)
async def index():return FileResponse(STATIC/'index.html')

@app.get('/{path:path}',include_in_schema=False)
async def spa(path:str):
 # Restrict served assets. Unknown routes go to SPA without exposing state or server files.
 if path.startswith(('api/','server/','state/')):raise HTTPException(404)
 return FileResponse(STATIC/'index.html')
