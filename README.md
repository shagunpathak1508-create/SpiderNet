# 🕷️ SpiderNet – Detect. Connect. Predict.
**Project Overview**

SpiderNet is an intelligent crime analysis and prediction system inspired by how spiders detect vibrations in their webs and sense disturbances. Instead of reacting after crimes occur, SpiderNet focuses on early detection, pattern recognition, and predictive insights using real-world crime data. The system transforms static crime datasets into a dynamic, interconnected intelligence network, enabling better decision-making and faster response.

**Problem Statement**

Crime systems are reactive, not predictive → action starts after damage is done
Rising urban crime is worsened by delayed detection and manual analysis
Emerging hotspots go unnoticed until they escalate
No real-time intelligence → slow response, poor resource use
Dashboards show data, not connections → patterns stay hidden
Missed patterns today become crimes tomorrow

**Proposed Solution**

Spiders sense danger through vibrations in their web — not by sight, but by signal pulses
This allows instant identification of threats and precise response
SpiderNet adapts this mechanism to crime data by detecting: Crime spikes Pattern clusters Hotspot intensity
By analyzing these signals collectively, the system identifies patterns, emerging threats, and hidden connections in real time.
Result: Instead of static reports, SpiderNet provides a living, responsive crime ecosystem — just like a spider’s web.
A **Streamlit-based crime analysis platform** for San Francisco PD crime data.  
Built for the DataWeb Hackathon, SpiderNet helps law enforcement intelligence teams detect anomalies, hotspots, and patterns in crime data.

---

## Features

| Tab | Description |
|-----|-------------|
| 📈 **Weekly Spikes** | Z-score–based anomaly detection on weekly crime volumes |
| 🗺️ **District Map** | Interactive Folium map with pulse-ring animations and "crime web" spider lines |
| 📊 **Breakdown** | Category, day-of-week, and hour-level crime breakdowns |
| 🤖 **Clusters** | KMeans geographical hotspot clustering |
| 🔍 **Explainer** | Actionable intelligence panel with recommended patrol actions |

---

## Setup

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

## Tech Stack

- **Streamlit** — Dashboard framework
- **Pandas / NumPy** — Data wrangling
- **Matplotlib / Seaborn** — Static charting
- **Folium + streamlit-folium** — Interactive maps
- **scikit-learn** — KMeans clustering
- **SciPy** — Z-score statistical anomaly detection

---

## Dataset

[SF Crime Classification — Kaggle](https://www.kaggle.com/competitions/sf-crime/data)  
San Francisco Police Department incident records from 2003–2015.

---

*Built with ❤️ for the DataWeb Ideathon*
