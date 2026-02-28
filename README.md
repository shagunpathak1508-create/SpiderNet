# 🕷️ SpiderNet — Crime Intelligence Dashboard

A **Streamlit-based crime analysis platform** for San Francisco PD crime data.  
Built for the DataWeb Hackathon, SpiderNet helps law enforcement intelligence teams detect anomalies, hotspots, and patterns in crime data.

---

## 🚀 Features

| Tab | Description |
|-----|-------------|
| 📈 **Weekly Spikes** | Z-score–based anomaly detection on weekly crime volumes |
| 🗺️ **District Map** | Interactive Folium map with pulse-ring animations and "crime web" spider lines |
| 📊 **Breakdown** | Category, day-of-week, and hour-level crime breakdowns |
| 🤖 **Clusters** | KMeans geographical hotspot clustering |
| 🔍 **Explainer** | Actionable intelligence panel with recommended patrol actions |

---

## 📦 Setup

### 1. Clone the repo
```bash
git clone https://github.com/shagunpathak1508/SpiderNet.git
cd SpiderNet
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download the **San Francisco Crime Classification** dataset from [Kaggle](https://www.kaggle.com/competitions/sf-crime/data) and place `train.csv` in the `data/` folder:
```
data/
  train.csv
```

> ⚠️ The data files are **not included** in this repo due to their large size (~200 MB).

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
SpiderNet/
├── app.py                          # Main Streamlit dashboard
├── requirements.txt                # Python dependencies
├── notebooks/
│   ├── SpiderNet_Crime_Analysis.ipynb   # Full EDA notebook
│   ├── ArachneX_analysis.ipynb          # Supplementary analysis
│   ├── cluster_map.png
│   ├── district_ranking.png
│   ├── heatmap_day_hour.png
│   ├── weekly_crime_dashboard.png
│   └── ...
└── data/                           # (not tracked — add your CSVs here)
    └── train.csv
```

---

## 🛠️ Tech Stack

- **Streamlit** — Dashboard framework
- **Pandas / NumPy** — Data wrangling
- **Matplotlib / Seaborn** — Static charting
- **Folium + streamlit-folium** — Interactive maps
- **scikit-learn** — KMeans clustering
- **SciPy** — Z-score statistical anomaly detection

---

## 📊 Dataset

[SF Crime Classification — Kaggle](https://www.kaggle.com/competitions/sf-crime/data)  
San Francisco Police Department incident records from 2003–2015.

---

*Built with ❤️ for the DataWeb Ideathon*
