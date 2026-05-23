# Group 22: E-Commerce Purchase Prediction & MLOps Architecture

**Academic Project:** Master's Level Machine Learning Pipeline & Deployment
**Author:** Justus Izuchukwu Onuh

## Key Links
* **[Project Report & Dataset (Google Drive)](#)** *(...)*
* **[System Demonstration (YouTube)](#)** *(...)*
* **Live API Endpoint:** `https://e-commerce-purchase-prediction.onrender.com`

---

## Overview
This repository contains a Full-Stack Machine Learning architecture designed to predict e-commerce conversion probabilities in real-time. Moving beyond static Jupyter Notebooks, this system implements a complete **MLOps lifecycle** featuring a Python/FastAPI backend and a React/Next.js frontend storefront. 

The core predictive engine uses an **Explainable AI (XAI)** approach via a strictly tuned Decision Tree Classifier, allowing business stakeholders to understand the exact mathematical drivers behind customer purchasing behavior.

##  System Architecture

### 1. Machine Learning Pipeline (`Phase2_Model_Training.ipynb`)
* **Algorithm:** Decision Tree Classifier (chosen for Gini impurity transparency over black-box models).
* **Data Balancing:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to resolve extreme e-commerce class imbalance (~92% non-buyers / 8% buyers), resulting in a mathematically balanced 50/50 training distribution.
* **Hyperparameter Optimization:** Utilized `GridSearchCV` with 5-fold cross-validation (`max_depth=10`, `min_samples_split=10`) to strictly prevent overfitting to the synthetic data.
* **Evaluation:** Model evaluated on unseen test data yielding a **0.8360 ROC/AUC score** and a **76% Recall** for actual buyers. 

### 2. Backend MLOps Server (`main.py`)
* **Framework:** FastAPI (Python)
* **Functionality:** * Deserializes the `.joblib` model and `LabelEncoder` for sub-second latency inference.
  * Serves as a dynamic product database API (`/api/products`).
  * Hosts the live prediction engine (`/api/predict`) applying business logic thresholds (e.g., triggering a 10% discount code for users with 40-60% conversion probability).
  * Logs user feedback (`/api/feedback`) for future concept drift detection and automated retraining.

### 3. Frontend Dashboard (`page.tsx`)
* **Framework:** Next.js (React) & Recharts
* **Functionality:** * Fetches live e-commerce products from the cloud backend.
  * Connects directly to the AI prediction endpoint to display real-time purchase probabilities when items are added to the cart.
  * Features dedicated internal views for Sales Managers (KPIs) and Data Scientists (Live XAI Feature Importance & ROC evaluation).

---

## Local Installation & Setup

To reproduce this environment locally, follow the steps below.

### Prerequisites
* Python 3.9+
* Node.js 18+ & npm

### Backend Setup (FastAPI)
1. Navigate to the backend directory:
   ```bash
   cd "E-Commerce Purchase Prediction"
