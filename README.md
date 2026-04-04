# 🌍 World Development Measurement — Country Clustering

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Table of Contents
- [Business Objective](#-business-objective)
- [Problem Statement](#-problem-statement)
- [Solution Approach](#-solution-approach)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Models & Evaluation](#-models--evaluation)
- [Key Outcomes](#-key-outcomes)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)

---

## 🎯 Business Objective

Segment countries across the globe into meaningful clusters based on key **economic, health, demographic, and infrastructure indicators** — enabling policymakers, researchers, and organizations (e.g., NGOs, UN bodies) to identify patterns in global development, target interventions more effectively, and benchmark nations against peer groups.

---

## ❓ Problem Statement

Countries differ vastly in their development trajectories. Raw metrics like GDP, infant mortality, internet usage, and CO₂ emissions, viewed in isolation, fail to reveal the structural groupings that exist across nations. The challenge is to:

- **Identify natural groupings** of countries based on multi-dimensional development data.
- **Compare clustering approaches** (K-Means, DBSCAN, Hierarchical) to find the most meaningful segmentation.
- **Deploy an interactive application** where users can explore cluster assignments and profiles.

---

## 💡 Solution Approach

1. **Exploratory Data Analysis (EDA)** — Understand distributions, correlations, skewness, and outliers across 25 development indicators for countries worldwide.
2. **Data Preprocessing** — Clean currency/percentage symbols, handle missing values, aggregate by country, treat outliers via IQR capping.
3. **Feature Engineering** — Group-level aggregation (mean per country), column renaming for model compatibility, dropping low-signal features.
4. **Model Building** — Apply three clustering algorithms: K-Means, DBSCAN, and Agglomerative (Hierarchical) Clustering.
5. **Model Evaluation** — Compare models using Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index.
6. **Model Deployment** — Serialize the best model (K-Means) using `pickle` and expose it via a Streamlit web application.

---

## 📊 Dataset

| Property | Details |
|---|---|
| **File** | `World_development_mesurement.csv` |
| **Records** | Multi-year data aggregated per country |
| **Countries** | 190+ unique countries |
| **Features** | 25 development indicators |

### Key Features

| Category | Features |
|---|---|
| **Health** | Birth Rate, Infant Mortality Rate, Life Expectancy (Male/Female), Health Exp % GDP, Health Exp/Capita |
| **Economy** | GDP, Business Tax Rate, Lending Interest, Tourism Inbound/Outbound |
| **Infrastructure** | CO₂ Emissions, Energy Usage, Internet Usage, Mobile Phone Usage |
| **Demographics** | Population (0-14, 15-64, 65+, Total, Urban) |
| **Governance** | Ease of Business, Days to Start Business, Hours to do Tax |

---

## 🔄 Project Workflow

```
Raw Data
   │
   ▼
Data Cleaning
(Remove $, %, commas → cast to float)
   │
   ▼
Aggregation
(Group by Country → Mean)
   │
   ▼
Missing Value Imputation
(Fill NaN with 0)
   │
   ▼
EDA
(Correlation heatmap, Boxplots, Pie charts, GDP analysis)
   │
   ▼
Outlier Treatment
(IQR-based capping on selected columns)
   │
   ▼
Model Building
(K-Means | DBSCAN | Hierarchical Clustering)
   │
   ▼
Model Evaluation
(Silhouette | Calinski-Harabasz | Davies-Bouldin)
   │
   ▼
Best Model → Pickle
   │
   ▼
Streamlit App Deployment
```

---

## 🤖 Models & Evaluation

### K-Means Clustering
- Used the **Elbow Method** (WCSS) to determine optimal clusters.
- Optimal clusters: **3** (after outlier treatment).
- Metrics:

| Metric | Score |
|---|---|
| Silhouette Coefficient | Higher is better (≥ 0.5 is good) |
| Calinski-Harabasz Score | Higher is better |
| Davies-Bouldin Index | Lower is better |

### DBSCAN
- Parameters tuned: `eps=0.4`, `min_samples=5`.
- Applied both before and after outlier treatment.
- Useful for detecting noise points / outlier countries.

### Hierarchical (Agglomerative) Clustering
- Dendrogram plotted using **complete linkage + euclidean distance**.
- Tested with 2 and 3 clusters.
- Evaluation compared against K-Means.

### ✅ Best Model: **K-Means (3 Clusters)**
K-Means delivered the best balance of cluster separation and compactness across all three metrics. It was selected for deployment.

---

## 📈 Key Outcomes

- Countries were successfully segmented into **3 distinct development clusters** — broadly corresponding to developed, developing, and underdeveloped economies.
- Strong positive correlations found between **GDP ↔ CO₂ Emissions, Energy Usage, Tourism**.
- Strong negative correlations found between **Birth Rate ↔ Life Expectancy**.
- **Ease of Business** shows a negative relationship with **Internet Usage** — suggesting administrative burden in digitally lagging economies.
- The final K-Means model is serialized as `model_kmeans.pkl` and served via a Streamlit app for interactive exploration.

---

## 📁 Project Structure

```
world-development-clustering/
│
├── data/
│   └── World_development_mesurement.csv      # Raw dataset
│
├── notebooks/
│   └── World_Development_Measure_.ipynb      # Full EDA + Modelling notebook
│
├── app/
│   └── app.py                                # Streamlit deployment app
│
├── models/
│   └── model_kmeans.pkl                      # Saved K-Means model
│
├── docs/
│   └── Project_clustering.docx               # Project brief / business objective
│
├── requirements.txt                          # Python dependencies
├── .gitignore                                # Files to exclude from version control
└── README.md                                 # Project documentation (this file)
```

---

## 🛠 Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| Pandas, NumPy | Data manipulation |
| Matplotlib, Seaborn | Data visualization |
| Scikit-learn | Clustering models & evaluation |
| SciPy | Hierarchical clustering / dendrogram |
| Pickle | Model serialization |
| Streamlit | Web app deployment |
| Jupyter Notebook | EDA & model development |
