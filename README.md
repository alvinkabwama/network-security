### Network Security Project for Phishing Data
🛡️ Network Security ML Pipeline (End-to-End ML + MLOps Project)

A complete machine learning system for network intrusion detection — from raw data ingestion to cloud deployment.

🚀 1. Project Overview

This project is an end-to-end machine learning pipeline designed to classify network traffic as benign or malicious.
It demonstrates my practical skills as a:

Data Scientist

Machine Learning Engineer

MLOps Engineer

Unlike tutorials that stop at “train the model,” this repo shows the full production lifecycle:
ingestion → validation → transformation → training → evaluation → packaging → CI/CD → deployment → inference API.

🔍 2. Problem Statement

Modern network environments face continuous threats: phishing, domain abuse, and malicious traffic patterns.

The goal of this project is to:

Build a machine-learning system that detects suspicious network behavior based on structured traffic data.

The dataset contains features such as:

URL length

SSL certificate details

Domain age

Abnormality indicators

Traffic metadata

Label: 0 = safe, 1 = malicious

This frames a binary classification intrusion detection task.

🏗️ 3. Architecture Overview
High-Level System Architecture
                          ┌──────────────────────────┐
                          │      GitHub Repo         │
                          └───────┬──────────────────┘
                                  │ Push to main
                                  ▼
                      ┌──────────────────────────┐
                      │     GitHub Actions       │
                      │  (CI/CD Workflow)        │
                      └───────┬──────────▲──────┘
                              │ Build     │ Deploy
                              ▼           │
            ┌──────────────────────────────────────────┐
            │  Build Docker Image → Push to AWS ECR    │
            └────────────────┬─────────────────────────┘
                             │
                             ▼ Pull
                 ┌──────────────────────────┐
                 │ AWS EC2 (Docker Host)    │
                 │ Runs FastAPI Inference   │
                 └───────────┬──────────────┘
                             │
                             ▼
                    User uploads CSV →
             REST API returns predictions

📊 4. ML Pipeline Architecture
End-to-End ML Workflow
Raw Data (MongoDB)
        │
        ▼
┌──────────────────────┐
│ 1. Data Ingestion     │
└───────────┬──────────┘
            ▼
┌───────────────────────────┐
│ 2. Data Validation         │
└───────────┬───────────────┘
            ▼
┌──────────────────────────┐
│ 3. Data Transformation    │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│ 4. Model Training         │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│ 5. Artifact Storage       │
└──────────────────────────┘

🧩 5. Project Structure
network-security/
│
├── networksecurity/
│   ├── cloud/
│   ├── components/
│   ├── constant/
│   ├── entity/
│   ├── exception/
│   ├── logging/
│   ├── pipeline/
│   └── utils/
│
├── app.py
├── Dockerfile
├── data_schema/
└── README.md

🧪 6. Key Features
✔ Data Ingestion

Reads raw data from MongoDB → saves train/test sets.

✔ Data Validation

Ensures schema correctness + generates drift reports.

✔ Data Transformation

Imputes missing values, scales features, saves preprocessor.pkl.

✔ Model Training

GridSearchCV, StratifiedKFold, Logistic Regression, RF, AdaBoost, GBoost, KNN.

✔ S3 Sync

Uploads artifacts + trained models to S3 for versioning.

✔ FastAPI Inference

CSV upload → Prediction table → HTML output.

🐳 7. Docker & Deployment (High-Level)
GitHub → GitHub Actions → Docker Build
           │
           ▼
Push to AWS ECR → EC2 pulls → FastAPI live

🔁 8. CI/CD Pipeline (Simplified)

Code pushed → GitHub Actions starts.

Build + test code.

Docker image created.

Push to ECR.

EC2 automatically pulls + restarts container.

📦 9. Running Locally
pip install -r requirements.txt
python networksecurity/pipeline/training_pipeline.py
uvicorn app:app --port 8888

☁️ 10. Running in Docker
docker build -t network-security .
docker run -p 8888:8888 network-security

🎯 11. Future Improvements

Add explainability (SHAP)

Monitoring + alerting

Scheduled retraining

Better model registry

👤 Author

Kabwama Leonald Alvin
Machine Learning Engineer | MLOps | AWS | DevOps

5️⃣ Save the File

Press:

CTRL + S