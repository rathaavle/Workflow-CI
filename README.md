# Workflow-CI

## California Housing - CI Pipeline dengan MLflow Project

**Nama:** Athalie Aurora  
**Docker Hub:** https://hub.docker.com/r/rathaavle/california-housing-model

---

## Struktur Repository

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                  ← GitHub Actions CI
├── MLProject/
│   ├── modelling.py                ← Training script
│   ├── conda.yaml                  ← Environment dependencies
│   ├── MLProject                   ← MLflow Project config
│   ├── california_housing_preprocessing/
│   │   ├── train.csv
│   │   └── test.csv
│   └── docker_hub_link.txt         ← Link Docker Hub (auto-generated)
└── README.md
```

---

## GitHub Secrets yang Diperlukan

| Secret               | Nilai            |
| -------------------- | ---------------- |
| `DAGSHUB_TOKEN`      | Token DagsHub    |
| `DOCKERHUB_USERNAME` | rathaavle        |
| `DOCKERHUB_TOKEN`    | Token Docker Hub |

---

## Cara Trigger CI

1. Push ke branch `main`
2. Atau manual via GitHub Actions → "Run workflow"

## Tahapan CI

1. Checkout repository
2. Setup Python 3.12.7
3. Install dependencies
4. Run preprocessing (generate dataset)
5. Train model via `mlflow run`
6. Upload artefak ke GitHub
7. Build Docker image via `mlflow models build-docker`
8. Push ke Docker Hub

---

## Docker Hub

```bash
docker pull rathaavle/california-housing-model:latest
docker run -p 5001:8080 rathaavle/california-housing-model:latest
```
