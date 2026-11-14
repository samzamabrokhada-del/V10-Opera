# PRO-PITCH OMEGA V10 — Your private Chrome home-screen app

**Status:** Source is built and tested locally. It is not hosted yet and no real API key is connected. The site will be private behind your chosen password after deployment.

## Step 1 — Get free football data

On your Android, visit **https://www.football-data.org/client/register** and register for the Free tier. The service supplies an API token through its account process. Keep it in your own account. Do not put it in this ZIP or a GitHub commit. The free tier has 12 competitions, delayed scores, and 10 calls/minute. The included default scanner starts with three competitions: Premier League, La Liga and Bundesliga; expand after testing.

## Step 2 — Make a private code repository

Open **https://github.com/new?visibility=private** while logged into GitHub. Name the repository something like `omega-private` and choose **Private**. The repo must contain the unzipped contents of *this ZIP at its root*, meaning `render.yaml`, `requirements.txt`, `server/` and `site/` are top-level items. The ZIP itself must **not** be committed as the only file. If uploading folders on Android is difficult, connect the GitHub app in ChatGPT so we can work in your private repository when permissions allow.

## Step 3 — Deploy on Render, Free plan

Open **https://dashboard.render.com/**. Choose **New → Blueprint**, authorize access to your private GitHub repository, and select `render.yaml` from its root. The blueprint explicitly requests a **Free** Python web service and private login. On the first Blueprint creation screen enter these secrets **only into Render's environment settings**:

* `FOOTBALL_DATA_API_KEY` → your football-data.org token
* `OMEGA_APP_PASSWORD` → your own unique password, **at least 12 characters**

`OMEGA_SESSION_SECRET` is generated automatically by Render. Never put either secret in your repo or send it to anyone.

Render should display an HTTPS address like `https://private-omega-example.onrender.com`. Open it on your phone. You should see the **OMEGA private login**; after signing in, open Chrome's **⋮** menu and choose **Add to Home screen** or **Install app**. You can now launch it with the OMEGA icon instead of opening a browser tab manually. It is **not** placed on the Play Store.

## What works / limits

* The Python service requests real fixtures/results from supported leagues, builds V10 research forecasts from recorded completed matches, and locks its selected paper predictions before kickoff when running. No bookmaker account is required.
* Historical scores and research probabilities are not guaranteed betting edges; the app stays in **research-only / NO BET** mode. It does not invent odds, xG, lineups or injuries.
* **Render Free sleeps after 15 minutes of inactivity, restarts on a later request, and loses local files when it sleeps or redeploys.** That means permanent result history and reliable always-on background scanning/push are **not yet supported** by this free deployment. Paid always-on hosting plus durable storage would be needed for dependable 24/7 alerts.
* The website has a public internet address, but its pages and data are password-protected by an HttpOnly cookie. Only `/api/health` is reachable without logging in. Your private GitHub repository does not make the host address inaccessible; the password gate is what protects it.
* If Render fails to start, check its build/deploy logs. The server refuses to start when private login credentials are missing or weak.

Tests: `python -m pytest tests -q` (13 checks). Local test: `uvicorn main:app --app-dir server --host 127.0.0.1 --port 8000` after dependencies and env variables are configured.

Data credit: Football data provided by the Football-Data.org API, when connected.
