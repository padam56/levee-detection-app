# Levee Detection App

FastAPI + React app for levee defect detection with Human-in-the-Loop (HITL) annotation support.

## Structure

- `backend/` - FastAPI inference API
- `frontend/` - React UI
- `Dockerfile` - one-container deploy (UI + API)

## Run Locally

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

## Run with Docker (Recommended)

From `levee-detection-app/`:

```bash
docker build -t levee-detection-app .
docker run --rm -p 7860:7860 levee-detection-app
```

Open `http://localhost:7860`.

## Deploy

Use GitHub + Hugging Face Spaces (free Docker hosting):

1. Push this folder to GitHub.
2. Create a Hugging Face Space with `SDK = Docker`.
3. Connect the GitHub repo and deploy.

Live URL format: `https://<space-name>.hf.space`

Current live demo:

`https://huggingface.co/spaces/padam5699/levee-detection-app`

## API Endpoints

- `GET /health`
- `GET /models`
- `POST /infer/image`
- `GET /annotations`
- `POST /annotations`
