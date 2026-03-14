FROM node:20-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r ./backend/requirements.txt

COPY backend/app ./backend/app
COPY --from=frontend-build /frontend/dist ./frontend/dist

# Keep model files in repo root for self-contained deployment.
COPY sandboil_best_model.h5 ./sandboil_best_model.h5
COPY seepage_best_model.h5 ./seepage_best_model.h5

EXPOSE 7860
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
