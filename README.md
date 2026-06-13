# Rajshahi Load Forecast — Web App

A Streamlit web app that produces a 24-hour hourly electricity load forecast
for the Rajshahi distribution zone. Users pick a date from a calendar
(today / tomorrow / day after tomorrow, Asia/Dhaka) and get a chart, table,
and downloadable CSV.

## Folder contents

```
streamlit_app/
├── app.py              # the web app
├── requirements.txt    # Python dependencies for Streamlit Cloud
├── README.md           # this file
└── artifacts/          # model files — COPY FROM GOOGLE DRIVE (see below)
    ├── config.json
    ├── best_model.pkl       (or best_model.keras)
    ├── feat_scaler.pkl
    ├── target_scaler.pkl
    └── history_tail.csv
```

## One-time deployment (about 20 minutes)

### Step 1 — Put the artifacts in place
1. Open Google Drive → `MyDrive/nesco_load_forecast/`.
2. Download these 5 files: `config.json`, `best_model.pkl`,
   `feat_scaler.pkl`, `target_scaler.pkl`, `history_tail.csv`.
3. Copy them into the `artifacts/` folder here.

### Step 2 — Create a GitHub repository
1. Sign up / log in at https://github.com (free).
2. Click **New repository** → name it e.g. `rajshahi-load-forecast` →
   set it to **Private** → Create.
3. Click **uploading an existing file** and drag in everything from this
   `streamlit_app/` folder (app.py, requirements.txt, README.md, and the
   whole `artifacts/` folder with its 5 files).
4. Click **Commit changes**.

### Step 3 — Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in **with your GitHub account**.
2. Click **Create app** → "Deploy a public app from GitHub".
3. Pick your repository, branch `main`, main file path `app.py`.
4. Click **Deploy**. First build takes ~3–5 minutes.
5. You get a URL like `https://rajshahi-load-forecast.streamlit.app`.

> Note: with a private repo, Streamlit Cloud asks for permission to read it —
> grant it during sign-in.

### Step 4 — Set the login credentials (REQUIRED — app won't open without it)
The app has a **username + password login**. Credentials live in Streamlit's
encrypted Secrets box, never in the code or on GitHub.

1. On https://share.streamlit.io, open your deployed app.
2. Click the **⋮ menu → Settings → Secrets**.
3. Paste a `[users]` block (see `secrets.toml.example` for the format):
   ```toml
   [users]
   "zobair.buet@gmail.com" = "your-strong-password"
   "operator1"             = "another-password"
   ```
4. Click **Save**. The app reloads with the login active.
5. Share the **URL + each person's username/password** with your colleagues.

To add/remove a user or change a password later: edit Secrets and Save.
No code change, no redeploy — it takes effect immediately.

## Weekly update routine (after retraining in Colab)

1. Run `train.ipynb` in Colab with the latest demand CSVs (as usual).
   New artifacts are written to `MyDrive/nesco_load_forecast/`.
2. Download the 5 artifact files from Drive.
3. On GitHub, open your repo → `artifacts/` folder → **Add file →
   Upload files** → drag the 5 new files in → **Commit changes**
   (GitHub replaces the old ones automatically).
4. Streamlit Cloud detects the commit and redeploys in ~1 minute.
   The app sidebar shows the new "Trained" date so you can confirm.

## Troubleshooting

- **"Could not load model artifacts"** — the `artifacts/` folder is missing
  one of the 5 files, or it wasn't uploaded to GitHub.
- **Pickle/version error loading the model** — Colab's library versions moved
  ahead of `requirements.txt`. Check what versions train.ipynb cell 1 prints,
  update `requirements.txt` to match, commit.
- **A DL model (BiGRU etc.) won training** — uncomment the `tensorflow-cpu`
  line in `requirements.txt` and upload `best_model.keras` instead of
  `best_model.pkl`.
- **App sleeps after inactivity** (free tier) — the first visitor after a
  quiet period waits ~1 minute while it wakes. Normal, no action needed.
- **Sidebar warns "History is N days old"** — retrain with fresh data; the
  longer the gap, the more hours are filled by seasonal-naive estimates.
