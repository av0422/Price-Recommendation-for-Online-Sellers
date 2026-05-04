import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Price Recommendation", layout="wide")

st.markdown("""
<style>
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pop {
    0%   { transform: scale(0.88); opacity: 0; }
    100% { transform: scale(1);    opacity: 1; }
}
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 16px 20px;
    animation: fadeUp 0.45s ease forwards;
}
div[data-testid="stMetricValue"] > div { color: #e94560 !important; font-size: 2rem !important; }
div[data-testid="stMetricLabel"]       { color: #aaa !important; }
</style>
""", unsafe_allow_html=True)

st.title("Price Recommendation for Online Sellers")
st.caption("Train ML models on retail data and get live AI-powered price predictions.")


# ── Sample dataset generators ─────────────────────────────────────────────────

def _make_df(stock_codes, descriptions, base_prices, countries, n=2000,
             qty_range=(1, 60), price_noise=0.18, qty_price_effect=-0.008, seed=42):
    rng = np.random.default_rng(seed)
    si = rng.integers(0, len(stock_codes), n)
    rows = []
    for i in range(n):
        s = si[i]
        qty = int(rng.integers(*qty_range))
        month = int(rng.integers(1, 13))
        price = max(0.1, base_prices[s] * (1 + rng.normal(0, price_noise)) + qty * qty_price_effect)
        rows.append({
            "InvoiceNo":   f"A{500000 + i}",
            "StockCode":   stock_codes[s],
            "Description": descriptions[s],
            "Quantity":    qty,
            "InvoiceDate": pd.Timestamp(f"2023-{month:02d}-{rng.integers(1, 28):02d}"),
            "UnitPrice":   round(price, 2),
            "CustomerID":  int(rng.integers(10000, 20000)),
            "Country":     countries[int(rng.integers(0, len(countries)))],
        })
    return pd.DataFrame(rows)


def gen_uk_retail():
    return _make_df(
        stock_codes=  ["85123A","71053","84406B","84029G","22752","21730","85099B","21212","22633","20725"],
        descriptions= ["White Hanging Heart","White Metal Lantern","Cream Cupid Hearts Coat Hanger",
                       "Knitted Union Flag Hot Water Bottle","Set of 3 Butterfly Wall Stickers",
                       "Glass Star Frosted T-Light Holder","Jumbo Bag Red Retrospot",
                       "Pack of 72 Retrospot Cake Cases","Hand Warmer Union Jack","Lunch Bag Red Retrospot"],
        base_prices=  [2.55, 3.39, 2.75, 3.39, 4.25, 5.95, 2.08, 0.42, 2.10, 1.65],
        countries=    ["United Kingdom","Germany","France","EIRE","Spain","Netherlands","Belgium","Switzerland"],
        qty_range=(1, 100), price_noise=0.15, qty_price_effect=-0.005,
    )

def gen_electronics():
    return _make_df(
        stock_codes=  ["EL001","EL002","EL003","EL004","EL005","EL006","EL007","EL008","EL009","EL010","EL011","EL012"],
        descriptions= ["USB-C Fast Charger","Wireless Mouse","Mechanical Keyboard","HD Webcam",
                       "Phone Stand Adjustable","Tempered Glass Screen Protector","20000mAh Power Bank",
                       "Smart Watch Pro","True Wireless Earbuds","7-Port USB Hub","LED Strip 5m","Gaming Headset"],
        base_prices=  [18, 32, 75, 55, 12, 8, 45, 159, 69, 28, 22, 89],
        countries=    ["United States","United Kingdom","Germany","Japan","Canada","Australia","France","Singapore"],
        qty_range=(1, 12), price_noise=0.12, qty_price_effect=-0.4,
    )

def gen_fashion():
    return _make_df(
        stock_codes=  ["FA001","FA002","FA003","FA004","FA005","FA006","FA007","FA008","FA009","FA010"],
        descriptions= ["Classic Cotton T-Shirt","Slim Fit Jeans","Floral Summer Dress","Leather Oxford Shoes",
                       "Merino Wool Scarf","Denim Jacket","Road Running Shoes","Silk Blouse","Cargo Trousers",
                       "Cable Knit Sweater"],
        base_prices=  [18, 55, 42, 95, 28, 68, 85, 72, 48, 60],
        countries=    ["France","Italy","United Kingdom","Spain","United States","Germany","Netherlands","Sweden"],
        qty_range=(1, 8), price_noise=0.20, qty_price_effect=-0.8,
    )

def gen_grocery():
    return _make_df(
        stock_codes=  ["GR001","GR002","GR003","GR004","GR005","GR006","GR007","GR008","GR009","GR010"],
        descriptions= ["Organic Whole Milk 2L","Sourdough Bread Loaf","Free Range Eggs 12pk",
                       "Greek Yogurt 500g","Mature Cheddar 400g","Fresh Orange Juice 1L",
                       "Brown Basmati Rice 1kg","Extra Virgin Olive Oil 500ml",
                       "Sparkling Mineral Water 6pk","70% Dark Chocolate 100g"],
        base_prices=  [1.45, 2.20, 2.80, 1.90, 2.50, 1.75, 1.60, 4.50, 3.20, 1.80],
        countries=    ["United Kingdom","Ireland","France","Germany","Netherlands","Belgium"],
        qty_range=(1, 25), price_noise=0.10, qty_price_effect=-0.02,
    )


DATASETS = {
    "UK Online Retail":    {"desc": "Gift & home décor e-commerce · 2,000 transactions",    "fn": gen_uk_retail},
    "Electronics & Tech":  {"desc": "Consumer electronics & gadgets · 2,000 transactions",  "fn": gen_electronics},
    "Fashion & Apparel":   {"desc": "Clothing & accessories retailer · 2,000 transactions", "fn": gen_fashion},
    "Grocery & FMCG":      {"desc": "Supermarket staples · 2,000 transactions",             "fn": gen_grocery},
}


# ── Training ──────────────────────────────────────────────────────────────────

def _train(df):
    df = df.dropna(subset=["Description"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[df["UnitPrice"] < df["UnitPrice"].quantile(0.99)]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Month"] = df["InvoiceDate"].dt.month

    le_stock   = LabelEncoder()
    le_country = LabelEncoder()
    df["StockCode_enc"] = le_stock.fit_transform(df["StockCode"].astype(str))
    df["Country_enc"]   = le_country.fit_transform(df["Country"].astype(str))

    features = ["Quantity", "StockCode_enc", "Country_enc", "Month"]
    X, y = df[features], df["UnitPrice"]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model_defs = {
        "Linear Regression": LinearRegression(),
        "Ridge":             Ridge(),
        "Lasso":             Lasso(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    trained, scores = {}, {}
    for name, mdl in model_defs.items():
        mdl.fit(X_train, y_train)
        scores[name]  = round(mdl.score(X_test, y_test), 4)
        trained[name] = mdl

    price_max   = float(y.quantile(0.95))
    stock_codes = sorted(df["StockCode"].astype(str).unique())
    countries   = sorted(df["Country"].unique())
    return trained, scaler, le_stock, le_country, scores, stock_codes, countries, price_max


@st.cache_data
def load_and_train_file(file_bytes, file_name):
    import io
    buf = io.BytesIO(file_bytes)
    df  = pd.read_excel(buf) if file_name.endswith(".xlsx") else pd.read_csv(buf, encoding="latin-1")
    return _train(df)


@st.cache_data
def load_and_train_sample(dataset_name):
    return _train(DATASETS[dataset_name]["fn"]())


# ── Data source UI ────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"],
    help="Expected columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country",
)

st.markdown("**— or explore a built-in sample dataset —**")
cols = st.columns(len(DATASETS))
chosen_sample = None
for col, (name, meta) in zip(cols, DATASETS.items()):
    with col:
        if st.button(f"**{name}**\n\n{meta['desc']}", key=name, use_container_width=True):
            st.session_state["sample"] = name
        if st.session_state.get("sample") == name:
            st.success(f"✓ Active")

if uploaded_file:
    st.session_state.pop("sample", None)

data_ready = False

if uploaded_file:
    with st.spinner("Training models on your file…"):
        result = load_and_train_file(uploaded_file.read(), uploaded_file.name)
    st.success("Trained on your dataset.")
    data_ready = True
elif st.session_state.get("sample"):
    chosen_sample = st.session_state["sample"]
    with st.spinner(f"Loading **{chosen_sample}** and training models…"):
        result = load_and_train_sample(chosen_sample)
    st.info(f"Using sample: **{chosen_sample}** — {DATASETS[chosen_sample]['desc']}")
    data_ready = True

# ── Main app ──────────────────────────────────────────────────────────────────

if data_ready:
    models, scaler, le_stock, le_country, scores, stock_codes, countries, price_max = result

    # Model performance
    st.subheader("Model Performance (R² Score)")
    score_df   = pd.DataFrame(scores.items(), columns=["Model", "R²"])
    fig_scores = px.bar(
        score_df, x="Model", y="R²", color="R²",
        color_continuous_scale="Blues", range_y=[0, 1], text_auto=".3f",
    )
    fig_scores.update_traces(textposition="outside")
    fig_scores.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        transition={"duration": 500, "easing": "cubic-in-out"},
    )
    st.plotly_chart(fig_scores, use_container_width=True)
    with st.expander("How to read this chart"):
        st.markdown("""
**R² (R-squared)** measures how well a model explains price variation in the test data.

| Score | Meaning |
|---|---|
| **1.0** | Perfect — model predicts every price exactly |
| **0.7 – 0.9** | Strong — most variation explained |
| **0.4 – 0.7** | Moderate — useful but noisy |
| **< 0.4** | Weak — model struggles with this data |

A higher bar = a more accurate model. Use the best-scoring model for your predictions below.
""")

    st.divider()

    # ── Prediction inputs ────────────────────────────────────────────────────
    st.subheader("Live Price Recommendation")
    col_in, col_gauge, col_table = st.columns([1.2, 1, 1], gap="large")

    with col_in:
        st.markdown("#### Inputs")
        selected_model = st.selectbox("Model", list(models.keys()))
        stock_code     = st.selectbox("Stock Code", stock_codes)
        country        = st.selectbox("Country", countries)
        quantity       = st.slider("Quantity", 1, 200, 10)
        month          = st.slider("Month", 1, 12, 6, help="1 = Jan, 12 = Dec")

    def _enc(le, val):
        try:    return le.transform([val])[0]
        except: return 0

    def predict_price(qty, stock, ctry, mo, model_name=None):
        mdl = models[model_name or selected_model]
        inp = scaler.transform([[qty, _enc(le_stock, stock), _enc(le_country, ctry), mo]])
        return mdl.predict(inp)[0]

    predicted = predict_price(quantity, stock_code, country, month)

    # Animated gauge
    with col_gauge:
        st.markdown("#### Predicted Price")
        gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = round(predicted, 2),
            number= {"prefix": "£", "font": {"size": 44, "color": "#e94560"}},
            gauge = {
                "axis":  {"range": [0, price_max * 1.2], "tickprefix": "£"},
                "bar":   {"color": "#1f77b4"},
                "steps": [
                    {"range": [0,               price_max * 0.4], "color": "#d4efdf"},
                    {"range": [price_max * 0.4, price_max * 0.8], "color": "#fdebd0"},
                    {"range": [price_max * 0.8, price_max * 1.2], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line":  {"color": "#e94560", "width": 3},
                    "thickness": 0.8,
                    "value": predicted,
                },
            },
        ))
        gauge.update_layout(
            margin={"t": 20, "b": 10, "l": 20, "r": 20},
            height=260,
            paper_bgcolor="rgba(0,0,0,0)",
            transition={"duration": 400, "easing": "cubic-in-out"},
        )
        st.plotly_chart(gauge, use_container_width=True)

    # All-models comparison table
    with col_table:
        st.markdown("#### All Models")
        rows = []
        for name in models:
            p = predict_price(quantity, stock_code, country, month, name)
            rows.append({"Model": name, "Price": f"£{p:.2f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with st.expander("How to read the gauge & table"):
        st.markdown("""
**Gauge chart** — the needle points to the predicted price for your selected model and inputs.

- **Green zone** — low end of the price range for this dataset
- **Orange zone** — mid range
- **Red zone** — high end (premium pricing territory)

The red line marks the exact predicted value.

**All Models table** — shows what every model would recommend for the same inputs.
If most models agree, you can be more confident in the price. A big spread means the models are uncertain — try adjusting quantity or checking if the stock code has enough training data.
""")

    st.divider()

    # ── Price × Quantity curve ───────────────────────────────────────────────
    st.subheader("Price vs Quantity Curve")
    qs    = list(range(1, 201))
    preds = [predict_price(q, stock_code, country, month) for q in qs]

    fig_line = px.line(
        x=qs, y=preds,
        labels={"x": "Quantity", "y": "Predicted Price (£)"},
        title=f"{selected_model} · {stock_code} · {country} · Month {month}",
    )
    fig_line.update_traces(line={"color": "#1f77b4", "width": 2.5})
    fig_line.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        transition={"duration": 400},
    )
    st.plotly_chart(fig_line, use_container_width=True)
    with st.expander("How to read this chart"):
        st.markdown("""
**X-axis** — quantity ordered (1 to 200 units).
**Y-axis** — the model's predicted unit price at that quantity.

- A **downward slope** means bulk buyers get a lower price (volume discount effect learned from data).
- A **flat line** means the model found no strong relationship between quantity and price.
- A **sharp drop** early on suggests the biggest discount happens at low quantities.

Use this to decide your pricing tiers — e.g. where to set a "buy 10+ and save" threshold.
""")

    st.divider()

    # ── Price heatmap: Country × Month ───────────────────────────────────────
    st.subheader("Price Heatmap — Country × Month")
    st.caption("Predicted price across every country and month for the selected stock code and quantity.")

    heat_countries = countries[:12]
    heat_data = np.array([
        [predict_price(quantity, stock_code, c, m) for m in range(1, 13)]
        for c in heat_countries
    ])
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig_heat = go.Figure(go.Heatmap(
        z=heat_data,
        x=month_labels,
        y=heat_countries,
        colorscale="Blues",
        text=[[f"£{v:.2f}" for v in row] for row in heat_data],
        texttemplate="%{text}",
        hovertemplate="Country: %{y}<br>Month: %{x}<br>Price: %{text}<extra></extra>",
        colorbar={"title": "Price (£)"},
    ))
    fig_heat.update_layout(
        height=350,
        margin={"t": 20, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        transition={"duration": 400},
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    with st.expander("How to read this chart"):
        st.markdown("""
**Each cell** shows the predicted price for that country (row) and month (column), for your chosen stock code and quantity.

- **Darker blue** = higher predicted price
- **Lighter blue** = lower predicted price
- **Hover** over any cell to see the exact value

Look across a row to spot seasonal patterns for a country — does price peak in December? Look down a column to compare how the same month prices differ by region. Use this to plan geo-targeted or seasonal pricing campaigns.
""")

    st.divider()

    # ── 3D Price Surface: Quantity × Month ───────────────────────────────────
    st.subheader("3D Price Surface — Quantity × Month")
    st.caption("Rotate and zoom to explore how quantity and season jointly affect price.")

    qty_range   = list(range(1, 101, 5))
    month_range = list(range(1, 13))
    Z = np.array([
        [predict_price(q, stock_code, country, m) for m in month_range]
        for q in qty_range
    ])
    fig_3d = go.Figure(go.Surface(
        x=month_labels,
        y=qty_range,
        z=Z,
        colorscale="Blues",
        hovertemplate="Month: %{x}<br>Quantity: %{y}<br>Price: £%{z:.2f}<extra></extra>",
    ))
    fig_3d.update_layout(
        scene={
            "xaxis_title": "Month",
            "yaxis_title": "Quantity",
            "zaxis_title": "Price (£)",
            "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 0.8}},
        },
        height=480,
        margin={"t": 10, "b": 10, "l": 10, "r": 10},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    with st.expander("How to read this chart"):
        st.markdown("""
**X-axis** — month of sale (Jan to Dec).
**Y-axis** — quantity ordered.
**Z-axis (height)** — predicted unit price.

- **Peaks** (high points) = combinations of quantity + month where the model predicts higher prices.
- **Valleys** (low points) = where prices are predicted to be lowest.
- **Click and drag** to rotate the surface. **Scroll** to zoom in/out.

This is the most powerful view for spotting interaction effects — e.g. maybe small orders in December command a premium, while large orders in summer are cheapest. Use it to find the sweet spot for your pricing strategy.
""")

else:
    st.info("Upload your dataset above — or click one of the sample dataset buttons to see the app in action.")
    st.markdown("""
**Expected columns:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`

Real dataset: [Online Retail on Kaggle](https://www.kaggle.com/datasets/vijayuv/onlineretail)
""")
