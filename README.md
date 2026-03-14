---
title: Levee Detection App
emoji: "robot"
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
---

# Levee Detection App (FastAPI + React)

Self-contained production-style app with:

- `backend/`: FastAPI inference API
- `frontend/`: React + Vite UI
- `Dockerfile`: single-container deploy (serves UI + API together)
- `docker-compose.yml`: optional split local stack

Model files are included at project root:

- `sandboil_best_model.h5`
- `seepage_best_model.h5`

## Local Development

Backend:

```bash
cd backend
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Single-Container Run (Prod-Like)

From this folder:

```bash
docker build -t levee-detection-app .
docker run --rm -p 7860:7860 levee-detection-app
```

Open `http://localhost:7860`.

## GitHub Push (Only This Folder)

From `levee-detection-app/`:

```bash
git init
git add .
git commit -m "Initial levee detection app"
git branch -M main
git remote add origin https://github.com/<your-username>/levee-detection-app.git
git push -u origin main
```

## Free Deployment (GitHub-Connected)

Recommended: Hugging Face Spaces (Docker)

1. Push this folder to a GitHub repo.
2. In Hugging Face, create a new Space:
3. `SDK`: `Docker`
4. Connect/select your GitHub repo.
5. Keep `app_port` as `7860`.
6. Deploy.

Your app will be available at:

`https://<your-space-name>.hf.space`

## API Endpoints

- `GET /health`
- `GET /models`
- `POST /infer/image`
- `GET /annotations`
- `POST /annotations`
