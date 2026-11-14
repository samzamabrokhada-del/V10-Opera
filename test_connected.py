import os,sys,copy,json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'server'))
import main,live_engine

NOW=datetime(2026,11,14,12,tzinfo=timezone.utc)

def match(i,ko,home,away,hg=None,ag=None,league='PL'):
 return {'id':'fd:'+str(i),'league':league,'league_name':'Premier League','season':'2026',
 'kickoff':ko.isoformat(),'home':home,'away':away,'status':'FINISHED' if hg is not None else 'SCHEDULED',
 'hg':hg,'ag':ag,'source':'football-data.org','timing_verified':True}

def history(n=55):
 teams=['Arsenal','Chelsea','Liverpool','Everton','Fulham','Wolves']
 rows=[]
 for i in range(n):
  a=teams[i%6];b=teams[(i+1+(i//6)%4)%6]
  if a==b:b=teams[(i+2)%6]
  k=NOW-timedelta(days=10+i*2)
  rows.append(match(i,k,a,b,(i+1)%4,(i*2)%3))
 return rows

def test_normalizer_matches_and_goals():
 doc={'matches':[{'id':44,'utcDate':'2026-11-18T21:00:00Z','homeTeam':{'name':'A'},'awayTeam':{'name':'B'},'status':'SCHEDULED','score':{'fullTime':{'home':None,'away':None}}},
 {'id':45,'utcDate':'2026-11-12T21:00:00Z','homeTeam':{'name':'C'},'awayTeam':{'name':'D'},'status':'FINISHED','score':{'fullTime':{'home':2,'away':1}}}]}
 x=live_engine.normalize_football_data('PL',doc)
 assert x[0]['hg'] is None and x[1]['hg']==2
 assert x[0]['id']=='fd:44' and x[0]['timing_verified']

def test_api_sports_normalizer():
 doc={'response':[{'fixture':{'id':55,'date':'2026-11-14T20:00:00Z','status':{'short':'NS'}},'teams':{'home':{'name':'A'},'away':{'name':'B'}},'league':{'id':5,'name':'Test League','season':2026},'goals':{'home':None,'away':None}}]}
 assert live_engine.normalize_api_sports(doc)[0]['league']=='AS5'

def test_rank_global_daily_cap_and_pre_match_only():
 fs=[]
 for i in range(10):
  f={'id':str(i),'kickoff':(NOW+timedelta(days=1,hours=i/8)).isoformat(),
   'date':'2026-11-15','probability':.96-i*.01,'forecast_at':NOW.isoformat(),'market':'1X'}
  fs.append(f)
 fs.append({'id':'past','kickoff':(NOW-timedelta(hours=1)).isoformat(),'probability':.99,'forecast_at':NOW.isoformat(),'market':'1X'})
 s=live_engine.rank_select(fs,now=NOW)
 assert len(s)==5 and 'past' not in s
 assert list(s)==['0','1','2','3','4']
 assert live_engine.rank_select(fs,s,now=NOW)==s

def test_analyze_never_rewrites_locked_prediction():
 data=history()
 data.append(match(200,NOW+timedelta(days=2),'Arsenal','Chelsea'))
 prev={'fd:200':{'id':'fd:200','kickoff':(NOW+timedelta(days=2)).isoformat(),'market':'X2','probability':.75,'date':'2026-11-16','forecast_at':NOW.isoformat()}}
 out=live_engine.analyze(data,prior_forecasts=prev,now=NOW)
 assert out['forecasts']['fd:200']==prev['fd:200']
 assert out['selected']=={}

def test_auto_settlement_correct_and_loss():
 future=(NOW-timedelta(days=1)).isoformat()
 past={'id':'fd:999','kickoff':future,'market':'X2','probability':.85,'date':'2026-11-13','forecast_at':(NOW-timedelta(days=2)).isoformat()}
 result=live_engine.analyze([match(999,NOW-timedelta(days=1),'A','B',0,2)],prior_forecasts={'fd:999':past},prior_selected={'fd:999':{'date':'2026-11-13','market':'X2','probability':.85,'locked_at':past['forecast_at']}},now=NOW)
 assert result['results'][0]['won'] is True and result['results'][0]['selected'] is True
 assert 'fd:999' in result['selected']

def test_provider_secrets_never_sent(monkeypatch,tmp_path):
 monkeypatch.setattr(main,'FD_KEY','private-demo-key')
 monkeypatch.setattr(main,'DATA_PATH',tmp_path/'snapshot.json')
 client=TestClient(main.app)
 res=client.get('/api/snapshot')
 assert res.status_code==200
 assert 'private-demo-key' not in res.text
 assert res.json()['configured']['football_data'] is True

def test_remote_credentials_required_for_refresh(monkeypatch):
 monkeypatch.setattr(main,'ADMIN_TOKEN','secret')
 assert TestClient(main.app).post('/api/refresh').status_code==403

def test_server_serves_connected_pwa():
 client=TestClient(main.app)
 for path in ['/','/live-ui.js','/manifest.webmanifest','/sw.js','/api/health']:
  assert client.get(path).status_code==200,path
 assert 'live-ui.js' in client.get('/').text
 assert 'OMEGA' in client.get('/').text

def test_empty_provider_does_not_create_fixtures(monkeypatch,tmp_path):
 monkeypatch.setattr(main,'DATA_PATH',tmp_path/'snap.json')
 monkeypatch.setattr(main,'FD_KEY','')
 monkeypatch.setattr(main,'AS_KEY','')
 st=main.sync_once()
 assert st['fixture_count']==0 and st['updated_at'] is None

def test_full_provider_to_model_sync(monkeypatch,tmp_path):
 import httpx
 monkeypatch.setattr(main,'DATA_PATH',tmp_path/'sync.json')
 monkeypatch.setattr(main,'FD_KEY','fake-key-for-mocked-connection')
 monkeypatch.setattr(main,'AS_KEY','')
 monkeypatch.setattr(main,'LEAGUES',['PL'])
 monkeypatch.setattr(main,'REQUEST_SPACING',0)
 monkeypatch.setattr(main,'ODDS_ENABLED',False)
 base=datetime.now(timezone.utc)
 teams=['Arsenal','Chelsea','Liverpool','Everton','Fulham','Wolves']
 matches=[]
 for i in range(56):
  h=teams[i%6];a=teams[(i+1+(i//6)%3)%6]
  if a==h:a=teams[(i+2)%6]
  ko=(base-timedelta(days=(i+10))).isoformat()
  matches.append({'id':i+100,'utcDate':ko,'homeTeam':{'name':h},'awayTeam':{'name':a},'status':'FINISHED',
  'competition':{'name':'Premier League'},'score':{'fullTime':{'home':i%3+1,'away':(i*2)%4}}})
 matches.append({'id':9999,'utcDate':(base+timedelta(days=2)).isoformat(),'homeTeam':{'name':'Arsenal'},'awayTeam':{'name':'Chelsea'},'status':'SCHEDULED',
  'competition':{'name':'Premier League'},'score':{'fullTime':{'home':None,'away':None}}})
 def handle(request):
  assert request.headers.get('x-auth-token')=='fake-key-for-mocked-connection'
  return httpx.Response(200,json={'matches':matches})
 transport=httpx.MockTransport(handle)
 class Client(httpx.Client):
  def __init__(self):super().__init__(transport=transport)
 st=main.sync_once(client_factory=Client)
 assert st['fixture_count']==57
 assert st['updated_at']
 assert st['models'].get('PL',{}).get('trained_matches')>=30
 assert 'fd:9999' in st['forecasts']
 assert st['forecasts']['fd:9999']['market'] in ('1X','X2')
 assert st['forecasts']['fd:9999']['calibrated'] is False
 snap=main._public_data(st)
 assert snap['configured']['football_data'] is True and snap['research_only']
 assert snap['actual_betting_signals']==0
 assert 'fake-key-for-mocked-connection' not in json.dumps(snap)


def test_static_query_string_cannot_expose_backend_code():
 c=TestClient(main.app)
 r=c.get('/app.js?_fname=../server/main.py')
 assert r.status_code==200
 assert 'PRO-PITCH OMEGA V10 / Android-first' in r.text
 assert 'FOOTBALL_DATA_API_KEY' not in r.text


def test_private_site_password_and_api_access(monkeypatch):
 monkeypatch.setattr(main,'REQUIRE_LOGIN',True)
 monkeypatch.setattr(main,'APP_PASSWORD','super-secret-personal-omega-password')
 monkeypatch.setattr(main,'SESSION_SECRET','S'*48)
 c=TestClient(main.app)
 assert c.get('/',follow_redirects=False).status_code==303
 assert c.get('/app.js',follow_redirects=False).status_code==303
 assert c.get('/api/snapshot').status_code==401
 assert c.get('/api/push/public-key').status_code==401
 assert c.get('/api/health').status_code==200
 assert 'OMEGA V10' in c.get('/login').text
 assert c.post('/login',data={'password':'wrong'},follow_redirects=False).status_code==401
 response=c.post('/login',data={'password':'super-secret-personal-omega-password'},follow_redirects=False)
 assert response.status_code==303
 assert 'httponly' in response.headers['set-cookie'].lower()
 assert 'samesite=strict' in response.headers['set-cookie'].lower()
 assert c.get('/').status_code==200
 assert c.get('/api/snapshot').status_code==200
 assert c.post('/logout',follow_redirects=False).status_code==303
 assert c.get('/api/snapshot').status_code==401


def test_auth_misconfiguration_fails_closed(monkeypatch):
 monkeypatch.setattr(main,'REQUIRE_LOGIN',True)
 monkeypatch.setattr(main,'APP_PASSWORD','')
 monkeypatch.setattr(main,'SESSION_SECRET','')
 c=TestClient(main.app)
 assert c.get('/api/snapshot').status_code==401
 assert c.get('/',follow_redirects=False).status_code==303
