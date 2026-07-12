# Uploading this project to GitHub — step by step

## 1. Create the repo on GitHub

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `hotel-review-intelligence` (or whatever you like)
3. **Leave it empty** — do NOT check "Add a README" or "Add .gitignore" (you already have both locally; letting GitHub create its own causes a conflict when you push)
4. Visibility: **Public** (so judges can access it) or **Private** with judges added as collaborators — check what the hackathon submission actually needs; a public repo is simplest since your OneDrive folder link is the real judged deliverable anyway
5. Click **Create repository** — GitHub will show you a page with commands. Ignore it; use the ones below instead (they're tailored to a project that already has files).

## 2. One-time git setup (skip if you've used git before on this machine)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## 3. Initialize and push, from inside your project folder

```powershell
# from PowerShell, inside "Hotel Review Intelligence"
cd "Desktop\Hotel Review Intelligence"

git init
git add .
git commit -m "Initial commit: Hotel Review Intelligence Engine"
git branch -M main
git remote add origin https://github.com/<your-username>/hotel-review-intelligence.git
git push -u origin main
```

If prompted for credentials, GitHub no longer accepts your account password over HTTPS —
use a **Personal Access Token** instead:
GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) →
Generate new token → check the `repo` scope → paste it in place of your password when git
asks.

## 4. Verify what actually got uploaded

After pushing, refresh the GitHub page and confirm:
- `hotel_reviews.json` is **NOT** there (it's gitignored — see below for why)
- `data/processed/` is **NOT** there (gitignored — regeneratable)
- `docs/screenshots/*.jpg` **ARE** there (so your README images render on GitHub)
- The README renders with images, badges, and tables showing correctly on the repo's main page

## 5. Every time you make a further change

```powershell
git add .
git commit -m "describe what you changed"
git push
```

---

## Which files are excluded from the repo, and why

I already created a `.gitignore` that handles this automatically — you don't need to
manually delete anything. Here's exactly what it excludes and the reasoning, so you know
what to expect when you check the GitHub page:

| Excluded | Why |
|---|---|
| `hotel_reviews.json` (19 MB) | This is the **hackathon's raw dataset**, not your own content. It's large, easily re-obtained from the toolkit, and redistributing an organizer-provided dataset publicly isn't necessary or clearly yours to publish. The README tells anyone cloning the repo where to get it. |
| `data/processed/*` (~4.5 MB total) | Fully **regeneratable** by `python -m src.pipeline` in a couple of minutes. Committing generated/cached files bloats the repo and risks going stale vs. the code that produced them — standard practice is to gitignore build artifacts, not commit them. |
| `src/__pycache__/*` | Compiled Python bytecode, auto-regenerated on every run. Never belongs in git for any Python project. |
| `interview-prep/NOTES.md` | Your personal prep notes for answering judge questions. Not part of the deliverable, and a little odd to publish your own "expected questions" cheat sheet. |
| `goal.md` | A raw paste of the hackathon toolkit's requirements text — your own scratch copy, not project content. |
| `Hackathon Participant Toolkit....docx` (2.1 MB) | This is **Expedia's own document**, not something you authored — it's their content, not yours to redistribute in your repo. |
| `.streamlit/secrets.toml` (if you ever create one) | Where API keys/secrets would live if you add any later. Never commit secrets. |
| Hugging Face model cache folders | Multi-GB model weight downloads — never commit these, they're not source code. |

## What IS included, and why

| Included | Why |
|---|---|
| `app.py`, `src/*.py` | The actual source code — the whole point of the repo. |
| `README.md`, `LICENSE`, `.gitignore` | Standard repo hygiene. |
| `docs/*.md` (assumptions, limitations, solution summary, deck outline) | Required hackathon deliverables — judges should be able to read these directly on GitHub without downloading anything. |
| `docs/screenshots/*.jpg` | So the README's screenshots actually render on the GitHub page. |
| `DEMO_VIDEO_SCRIPT.md` | Shows your process/rigor, costs nothing to include. |
| `sample_output.json` | The schema reference your output is validated against — small and useful for anyone reading the code. |
| `user_profiles.json` | Small (12 KB) — kept so someone can clone and partially explore without having to source the big review file first. |
| `outputs/rich/*.json`, `outputs/submission/*.json` | Your **actual computed results** for all 50 profiles (~700 KB total). Judges can browse real output without running the pipeline themselves — high value, low size. |
| `notebook6eb4e2c4a1.ipynb` | A self-contained, reproducible notebook version of the pipeline (useful if a judge wants to run everything on Kaggle/Colab without setting up a local environment). |
| `data/sentence_overrides.csv` | Small config file — the manual audit corrections that back your "100%-audited" claim in the docs. Worth keeping visible as evidence of that process. |

If you'd rather include or exclude anything differently (e.g. you do want the raw
`hotel_reviews.json` in the repo for full reproducibility), just delete the matching line
from `.gitignore` before your first `git add .`.
