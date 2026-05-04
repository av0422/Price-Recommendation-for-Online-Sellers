# Price Recommendation for Online Sellers

An interactive machine learning web app that trains regression models on retail transaction data and recommends optimal prices for products — built with Streamlit and Plotly.

---

## Features

- **4 built-in sample datasets** — UK Online Retail, Electronics & Tech, Fashion & Apparel, Grocery & FMCG — no upload needed to explore
- **Upload your own CSV or Excel** retail dataset and train on real data
- **4 ML models** trained simultaneously: Linear Regression, Ridge, Lasso, Random Forest
- **Live price prediction** — adjust stock code, country, quantity, and month to get instant recommendations
- **Animated gauge chart** showing predicted price with visual risk zones
- **Country × Month heatmap** — see how price varies across all countries and seasons at once
- **3D price surface** — rotate and zoom to explore how quantity and month jointly affect price
- **Price vs Quantity curve** — understand bulk pricing dynamics per model

---

## Demo

| Section | Description |
|---|---|
| Sample dataset buttons | One-click load, no file needed |
| Model performance bar chart | Compare R² scores across all 4 models |
| Gauge chart | Animated predicted price with low/mid/high zones |
| Heatmap | Predicted price across every country × month combination |
| 3D surface | Interactive price landscape by quantity and month |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/price-recommendation-app.git
cd price-recommendation-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Using Your Own Dataset

Upload a CSV or Excel file with these columns:

| Column | Description |
|---|---|
| `InvoiceNo` | Transaction ID |
| `StockCode` | Product code |
| `Description` | Product name |
| `Quantity` | Units sold |
| `InvoiceDate` | Date of sale |
| `UnitPrice` | Price per unit |
| `CustomerID` | Customer identifier |
| `Country` | Country of sale |

A compatible real-world dataset: [Online Retail Dataset on Kaggle](https://www.kaggle.com/datasets/vijayuv/onlineretail)

---

## Tech Stack

| Library | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web app framework |
| [scikit-learn](https://scikit-learn.org) | ML models and preprocessing |
| [Plotly](https://plotly.com/python/) | Interactive and animated charts |
| [pandas](https://pandas.pydata.org) | Data manipulation |
| [NumPy](https://numpy.org) | Numerical computations |

---

## Deploying (Free)

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in
3. Click **New app** → select your repo → set main file to `app.py`
4. Click **Deploy**

Your app will be live at a public URL in under a minute.

---

## Project Structure

```
price_recommendation_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
```
