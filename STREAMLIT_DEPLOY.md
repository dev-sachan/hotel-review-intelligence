# Deploying to Streamlit Community Cloud — step by step

This deploys the app straight from your GitHub repo, free, with a public URL you can drop
into your Solution Summary and demo video. No server to manage.

## Prerequisites

- Your code is already pushed to GitHub (see `GITHUB_SETUP.md` if not done yet)
- Confirm `data/processed/` **is** actually in your GitHub repo — check the repo page in
  your browser. This is the one thing that's different from normal git hygiene: I
  deliberately kept these ~4.7MB cached files **in** git (not gitignored) specifically so
  the deployed app works instantly with zero setup. If they're missing from GitHub, the
  live app will show "cached artifacts not found" — go back and re-check your `.gitignore`
  and re-commit if needed.

## Steps

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub
   account (same account you pushed the repo to).
2. Click **"Create app"** → **"Deploy a public app from GitHub"**.
3. Fill in:
   - **Repository:** `<your-username>/hotel-review-intelligence`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Advanced settings"** before deploying (do this now, not after):
   - **Python version:** 3.11
   - You don't need to add any secrets — this app doesn't use API keys.
5. Click **Deploy**.

## What happens next

- First deploy takes a few minutes — Streamlit Cloud installs everything in
  `requirements.txt`, including a ~200MB CPU-only PyTorch build (kept small deliberately —
  see the comment in `requirements.txt`).
- The app itself starts instantly once installed, because it only reads the cached
  `data/processed/` files you committed — it never re-runs the NLP pipeline live.
- The **first time** someone uses Evidence Search (semantic search tab), there's a small
  ~80MB one-time download of the MiniLM embedding model from Hugging Face. This happens
  once per app "sleep cycle," not per user — Streamlit Cloud keeps the app warm while it
  has recent traffic.

## After it's live

You'll get a URL like:
```
https://hotel-review-intelligence-<random>.streamlit.app
```
(You can pick a custom subdomain in the app settings if you want something cleaner, e.g.
`hotel-review-intelligence.streamlit.app` if it's available.)

**Update these two places with the real link:**
1. `Solution_Summary.docx` — the yellow-highlighted "Live Demo" hyperlink near the top
2. `README.md` — the `[**🔗 Live Demo**]` badge link near the top

## If something goes wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| "Cached artifacts not found" error on the live app | `data/processed/` didn't actually get pushed to GitHub | Check `.gitignore` doesn't exclude it, `git add -f data/processed/`, commit, push again |
| Build fails / times out | `torch` installing the full CUDA wheel instead of CPU-only | Confirm the `--extra-index-url https://download.pytorch.org/whl/cpu` line is present and above `torch` in `requirements.txt` |
| Evidence Search shows a warning instead of results on first try | Normal — the embedding model is downloading in the background. Wait a few seconds and search again. | No action needed, this is expected on a cold start |
| App sleeps after inactivity | Streamlit Community Cloud free tier sleeps unused apps | Visit the link yourself a few minutes before your demo/judging so it's already "warm" |

## A note for judges/graders using the link

Streamlit Cloud's free tier puts an app to sleep after a period of no traffic, and the
first visitor after that wakes it back up (takes ~30-60 seconds). If you know roughly when
judges will click the link, open it yourself shortly beforehand so nobody hits a cold
start.
