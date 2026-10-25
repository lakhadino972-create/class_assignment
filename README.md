# Dialysis Need Predictor

Web app that predicts whether a patient may need dialysis from lab values and history.
Stack: scikit-learn model, FastAPI backend, plain HTML/CSS/JS frontend, Docker.

## Project structure
```
dialysis-predictor/
├── backend/
│   ├── app.py        # FastAPI API + serves frontend
│   └── train.py      # trains and saves the model
├── frontend/         # index.html, style.css, script.js
├── data/kidney_disease_dataset.csv
├── model/            # created by train.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python backend/train.py          # creates model/
uvicorn backend.app:app --reload
```
Open http://localhost:8000 (API docs at /docs).

## Run with Docker
```bash
docker compose up --build
```
Or: `docker build -t dialysis-app .` then `docker run -p 8000:8000 dialysis-app`.

## API
`POST /api/predict` with JSON:
```json
{"Age":58,"Creatinine_Level":2.9,"BUN":35,"GFR":11.5,
 "Urine_Output":600,"Diabetes":1,"Hypertension":1,"CKD_Status":1}
```
Other endpoints: `GET /api/health`, `GET /api/model-info`.

## Dataset notes
- 2,304 rows, only 31 (1.35%) are dialysis cases, so the model uses stratified
  splits and F1/recall instead of accuracy.
- In this dataset every dialysis case has GFR < 15 and CKD = 1, so the model mostly
  learns that pattern. The data looks synthetic; do not treat results as clinical.

## Disclaimer
Educational project only. Not medical advice or a diagnosis.
