import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, auc, calinski_harabasz_score,
                             confusion_matrix, davies_bouldin_score, f1_score,
                             precision_score, recall_score, roc_curve,
                             silhouette_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from mlxtend.frequent_patterns import apriori, association_rules

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ----------------------------------------------------------------------------
# Page setup & styling
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Restaurant Analytics", page_icon="🍽️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {background: linear-gradient(120deg, #ff6b35 0%, #f7931e 55%, #ffc15e 100%);
           padding: 22px 28px; border-radius: 14px; color: white; margin-bottom: 18px;}
    .hero h1 {margin: 0; font-size: 1.9rem; color: white;}
    .hero p {margin: 4px 0 0 0; opacity: .95;}
    div[data-testid="stMetric"] {background: rgba(255,107,53,.07); border: 1px solid rgba(255,107,53,.25);
           padding: 14px 16px; border-radius: 12px;}
    div[data-testid="stMetricLabel"] {font-weight: 600;}
    button[data-baseweb="tab"] {font-weight: 600;}
    .insight {background: rgba(46,134,171,.08); border-left: 4px solid #2e86ab;
              padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

COLORS = px.colors.qualitative.Bold
TEMPLATE = "plotly_white"


def _st_version():
    try:
        return tuple(int(p) for p in st.__version__.split(".")[:2])
    except ValueError:
        return (0, 0)


# Streamlit >= 1.50 uses width="stretch" instead of use_container_width.
STRETCH = {"width": "stretch"} if _st_version() >= (1, 50) else {"use_container_width": True}


def show(fig, height=None):
    fig.update_layout(template=TEMPLATE, margin=dict(l=10, r=10, t=55, b=10),
                      colorway=COLORS, title_font_size=16)
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, **STRETCH)


def table(data, **kwargs):
    st.dataframe(data, hide_index=True, **STRETCH, **kwargs)


def insight(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


DATA = Path(".")

FILES = {
    "customers": "restaurant-sample-database-v1-csv-customers.csv",
    "menu": "restaurant-sample-database-v1-csv-food_and_beverage_catalog.csv",
    "categories": "restaurant-sample-database-v1-csv-food_category.csv",
    "items": "restaurant-sample-database-v1-csv-order_items.csv",
    "orders": "restaurant-sample-database-v1-csv-orders.csv",
    "payments": "restaurant-sample-database-v1-csv-payments.csv",
}
REQUIRED = ["customers", "menu", "categories", "items", "orders"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ----------------------------------------------------------------------------
# Data loading (same logic as the original)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    missing = [FILES[k] for k in REQUIRED if not (DATA / FILES[k]).exists()]
    if missing:
        raise FileNotFoundError("Missing CSV files: " + ", ".join(missing))

    customers = pd.read_csv(DATA / FILES["customers"])
    menu = pd.read_csv(DATA / FILES["menu"])
    categories = pd.read_csv(DATA / FILES["categories"])
    items = pd.read_csv(DATA / FILES["items"])
    orders = pd.read_csv(DATA / FILES["orders"])

    df = items.merge(orders, on="order_id", suffixes=("_item", "_order"))
    df = df.merge(customers, on="customer_id", how="left", suffixes=("", "_customer"))
    df = df.merge(menu, on="item_id", how="left", suffixes=("", "_menu"))
    df = df.merge(categories, on="category_id", how="left", suffixes=("", "_category"))

    price_candidates = [c for c in df.columns
                        if c.lower() in ("price", "price_item", "price_menu")]
    if not price_candidates:
        raise ValueError("No price column found. Check the order_items and menu CSV headers.")
    price_col = "price_item" if "price_item" in df.columns else price_candidates[0]
    df["price"] = pd.to_numeric(df[price_col], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    date_candidates = [c for c in df.columns
                       if c.lower() in ("order_date", "date", "created_at")]
    if not date_candidates:
        raise ValueError("No order date column found. Check the orders CSV headers.")
    df["order_date"] = pd.to_datetime(df[date_candidates[0]], errors="coerce")

    df = df.dropna(subset=["order_id", "customer_id", "item_id",
                           "order_date", "quantity", "price"])
    df = df[(df["quantity"] > 0) & (df["price"] >= 0)]
    if "order_item_id" in df.columns:
        df = df.drop_duplicates(subset=["order_item_id"])
    df["sales_amount"] = df["quantity"] * df["price"]
    df["order_total"] = df.groupby("order_id")["sales_amount"].transform("sum")

    if "name" not in df.columns:
        name_cols = [c for c in df.columns if "name" in c.lower()]
        df["name"] = df[name_cols[0]] if name_cols else df["item_id"].astype(str)

    if "category_name" in df.columns:
        df["category"] = df["category_name"]
    elif "name_category" in df.columns:
        df["category"] = df["name_category"]
    elif "category" not in df.columns:
        df["category"] = "Unknown"
    df["category"] = df["category"].fillna("Unknown")
    return df.reset_index(drop=True)


try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load the datasets: {e}")
    st.info("Check the CSV filenames and column names in your project folder.")
    st.stop()

# ----------------------------------------------------------------------------
# Sidebar: filters
# ----------------------------------------------------------------------------
st.sidebar.title("🎛️ Filters")
min_date, max_date = df["order_date"].min().date(), df["order_date"].max().date()

date_range = st.sidebar.date_input("📅 Order date range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    st.sidebar.info("Select both a start and an end date. Showing the full range meanwhile.")
    start_date, end_date = min_date, max_date

all_categories = sorted(df["category"].unique())
sel_categories = st.sidebar.multiselect("🍴 Categories", all_categories, default=all_categories)
currency = st.sidebar.selectbox("💱 Currency symbol", ["", "$", "€", "£", "₹"], index=0)

scoped = df[df["category"].isin(sel_categories)]

with st.sidebar.expander("⚙️ Advanced filters"):
    sel_items = st.multiselect("Menu items (empty = all)", sorted(df["name"].unique()))
    sel_days = st.multiselect("Weekday", WEEKDAYS, default=WEEKDAYS)

    p_lo, p_hi = float(df["price"].min()), float(df["price"].max())
    if p_lo < p_hi:
        price_rng = st.slider("Unit price", p_lo, p_hi, (p_lo, p_hi))
        scoped = scoped[scoped["price"].between(*price_rng)]

    o_lo, o_hi = float(df["order_total"].min()), float(df["order_total"].max())
    if o_lo < o_hi:
        total_rng = st.slider("Order total", o_lo, o_hi, (o_lo, o_hi))
        scoped = scoped[scoped["order_total"].between(*total_rng)]

    q_lo, q_hi = int(df["quantity"].min()), int(df["quantity"].max())
    if q_lo < q_hi:
        qty_rng = st.slider("Quantity per line", q_lo, q_hi, (q_lo, q_hi))
        scoped = scoped[scoped["quantity"].between(*qty_rng)]

    # Optional attribute filters, shown only if these columns exist in your data.
    for base in ("gender", "city", "state", "country", "loyalty_tier", "membership",
                 "order_status", "status", "order_type", "channel"):
        for suffix in ("", "_customer", "_order"):
            col = base + suffix
            if col in scoped.columns and 2 <= scoped[col].nunique() <= 30:
                options = sorted(scoped[col].dropna().astype(str).unique())
                chosen = st.multiselect(col.replace("_", " ").title(), options)
                if chosen:
                    scoped = scoped[scoped[col].astype(str).isin(chosen)]
                break

if sel_items:
    scoped = scoped[scoped["name"].isin(sel_items)]
scoped = scoped[scoped["order_date"].dt.dayofweek.map(lambda i: WEEKDAYS[i]).isin(sel_days)]

filtered = scoped[scoped["order_date"].dt.date.between(start_date, end_date)].copy()

# Previous period of equal length (for KPI deltas)
span_days = (end_date - start_date).days + 1
prev_start = pd.Timestamp(start_date) - pd.Timedelta(days=span_days)
previous = scoped[(scoped["order_date"] >= prev_start)
                  & (scoped["order_date"] < pd.Timestamp(start_date))]

st.markdown(
    '<div class="hero"><h1>🍽️ Restaurant Sales & Customer Behavior Analysis</h1>'
    "<p>Sales analytics • Customer segmentation • Classification • Market basket analysis</p></div>",
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("No sales found for the selected filters. Try widening them.")
    st.stop()

st.sidebar.success(f"{len(filtered):,} order lines • {filtered['order_id'].nunique():,} orders")


def money(x):
    return f"{currency}{x:,.2f}"


def kpi_values(d):
    sales, orders = d["sales_amount"].sum(), d["order_id"].nunique()
    return {"sales": sales, "orders": orders, "customers": d["customer_id"].nunique(),
            "aov": sales / orders if orders else 0, "items": d["quantity"].sum()}


def delta(cur, prev):
    return f"{(cur - prev) / prev * 100:+.1f}% vs prev. period" if prev else None


cur = kpi_values(filtered)
prv = kpi_values(previous) if not previous.empty else None
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💰 Total Sales", money(cur["sales"]), delta(cur["sales"], prv and prv["sales"]))
c2.metric("🧾 Orders", f"{cur['orders']:,}", delta(cur["orders"], prv and prv["orders"]))
c3.metric("👥 Customers", f"{cur['customers']:,}", delta(cur["customers"], prv and prv["customers"]))
c4.metric("🛒 Avg Order Value", money(cur["aov"]), delta(cur["aov"], prv and prv["aov"]))
c5.metric("🍔 Items Sold", f"{cur['items']:,.0f}", delta(cur["items"], prv and prv["items"]))

overview, sales_tab, segment_tab, predict_tab, basket_tab, data_tab = st.tabs([
    "📊 Overview", "📈 Sales Analytics", "👥 Customer Segmentation",
    "🌳 Customer Classification", "🧺 Market Basket", "🗂️ Dataset",
])

# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
with overview:
    daily = (filtered.assign(day=filtered["order_date"].dt.normalize())
             .groupby("day", as_index=False)["sales_amount"].sum())
    daily["7-day average"] = daily["sales_amount"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["sales_amount"], name="Daily sales",
                             fill="tozeroy", line=dict(width=1.5), opacity=.6))
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["7-day average"], name="7-day average",
                             line=dict(width=3, color="#ff6b35")))
    fig.update_layout(title="Sales trend with 7-day moving average",
                      legend=dict(orientation="h", y=1.12))
    show(fig, 380)

    left, right = st.columns(2)
    with left:
        cat_sales = filtered.groupby("category", as_index=False)["sales_amount"].sum()
        show(px.pie(cat_sales, names="category", values="sales_amount", hole=.5,
                    title="Revenue share by category"), 380)
    with right:
        item_sales = (filtered.groupby("name", as_index=False)["sales_amount"].sum()
                      .nlargest(10, "sales_amount").sort_values("sales_amount"))
        show(px.bar(item_sales, x="sales_amount", y="name", orientation="h",
                    color="sales_amount", color_continuous_scale="Oranges",
                    title="Top 10 menu items by revenue"), 380)

    left, right = st.columns(2)
    with left:
        tree_df = filtered.groupby(["category", "name"], as_index=False)["sales_amount"].sum()
        show(px.treemap(tree_df, path=["category", "name"], values="sales_amount",
                        title="Revenue breakdown: category → item"), 400)
    with right:
        dow = (filtered.groupby(filtered["order_date"].dt.dayofweek)["sales_amount"].sum()
               .reindex(range(7), fill_value=0))
        show(px.bar(x=WEEKDAYS, y=dow.values, color=dow.values,
                    color_continuous_scale="Tealgrn", title="Sales by weekday",
                    labels={"x": "", "y": "Sales", "color": "Sales"}), 400)

# ----------------------------------------------------------------------------
# Sales analytics
# ----------------------------------------------------------------------------
with sales_tab:
    grain = st.radio("Time granularity", ["Daily", "Weekly", "Monthly"], index=2, horizontal=True)
    rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[grain]
    series = (filtered.set_index("order_date")["sales_amount"].resample(rule).sum().reset_index())
    series["growth"] = series["sales_amount"].pct_change() * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=series["order_date"], y=series["sales_amount"], name="Sales", opacity=.85)
    fig.add_scatter(x=series["order_date"], y=series["growth"], name="Growth %",
                    mode="lines+markers", line=dict(color="#d62728"), secondary_y=True)
    fig.update_layout(title=f"{grain} sales and period-over-period growth",
                      legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title_text="Sales", secondary_y=False)
    fig.update_yaxes(title_text="Growth %", secondary_y=True)
    show(fig, 400)

    left, right = st.columns(2)
    with left:
        order_level = (filtered.groupby("order_id")
                       .agg(date=("order_date", "min"), total=("sales_amount", "sum")))
        aov = order_level.set_index("date")["total"].resample("MS").mean().reset_index()
        show(px.line(aov, x="date", y="total", markers=True,
                     title="Average order value by month",
                     labels={"total": "Avg order value", "date": ""}), 360)
    with right:
        if (filtered["order_date"].dt.hour != 0).any():
            heat = (filtered.assign(dow=filtered["order_date"].dt.dayofweek,
                                    hour=filtered["order_date"].dt.hour)
                    .pivot_table(index="dow", columns="hour", values="sales_amount", aggfunc="sum")
                    .reindex(range(7)))
            heat.index = WEEKDAYS
            show(px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                           title="Sales heatmap: weekday × hour",
                           labels=dict(x="Hour", y="", color="Sales")), 360)
        else:
            show(px.histogram(order_level, x="total", nbins=30,
                              title="Distribution of order values",
                              labels={"total": "Order total"}), 360)

    st.subheader("Category performance")
    category_summary = (filtered.groupby("category", as_index=False)
                        .agg(Total_Sales=("sales_amount", "sum"), Items_Sold=("quantity", "sum"),
                             Orders=("order_id", "nunique"))
                        .sort_values("Total_Sales", ascending=False))
    category_summary["Share_%"] = category_summary["Total_Sales"] / category_summary["Total_Sales"].sum() * 100
    table(category_summary, column_config={
        "Share_%": st.column_config.ProgressColumn("Share %", min_value=0, max_value=100, format="%.1f%%"),
        "Total_Sales": st.column_config.NumberColumn(format="%.2f")})

    st.subheader("Top menu items")
    top_items = (filtered.groupby("name", as_index=False)
                 .agg(Total_Sales=("sales_amount", "sum"), Quantity_Sold=("quantity", "sum"),
                      Orders=("order_id", "nunique"))
                 .sort_values("Total_Sales", ascending=False).head(15))
    table(top_items, column_config={
        "Total_Sales": st.column_config.ProgressColumn(
            "Total sales", min_value=0, max_value=float(top_items["Total_Sales"].max()), format="%.2f")})

# ----------------------------------------------------------------------------
# Customer segmentation (K-Means with proper k selection)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Evaluating cluster counts…")
def evaluate_k(X, k_max):
    rows = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        labels = km.labels_
        if len(set(labels)) < 2:
            continue
        rows.append({
            "k": k,
            "Inertia": km.inertia_,
            "Silhouette": silhouette_score(X, labels, sample_size=min(len(X), 5000), random_state=42),
            "Davies-Bouldin": davies_bouldin_score(X, labels),
            "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        })
    return pd.DataFrame(rows)


def elbow_k(ks, inertias):
    """Point furthest below the straight line joining the first and last inertia."""
    x, y = np.array(ks, float), np.array(inertias, float)
    if len(x) < 3 or y.max() == y.min():
        return int(ks[0])
    xn, yn = (x - x.min()) / (x.max() - x.min()), (y - y.min()) / (y.max() - y.min())
    return int(ks[int(np.argmax((1 - xn) - yn))])


def label_clusters(profile_z):
    """Human-friendly names from standardized cluster means (low recency = recent)."""
    names = {}
    for cluster, row in profile_z.iterrows():
        if row["recency"] < 0 and row["monetary"] > 0:
            names[cluster] = "High-value active"
        elif row["recency"] >= 0 and row["monetary"] > 0:
            names[cluster] = "High-value at risk"
        elif row["recency"] < 0:
            names[cluster] = "Active, lower spend"
        else:
            names[cluster] = "Lapsed, low value"
    return names


with segment_tab:
    st.subheader("RFM Customer Segmentation")
    st.caption("Recency = days since last order; Frequency = distinct orders; Monetary = total sales.")

    latest_date = filtered["order_date"].max().normalize() + pd.Timedelta(days=1)
    rfm = filtered.groupby("customer_id").agg(
        last_order=("order_date", "max"),
        frequency=("order_id", "nunique"),
        monetary=("sales_amount", "sum"),
    ).reset_index()
    rfm["recency"] = (latest_date - rfm["last_order"]).dt.days

    if len(rfm) < 4:
        st.warning("At least 4 customers are needed to compare cluster counts.")
    else:
        use_log = st.checkbox("Log-transform features (recommended: spend and frequency are skewed)", value=True)
        feats = rfm[["recency", "frequency", "monetary"]].astype(float)
        if use_log:
            feats = np.log1p(feats)
        X = StandardScaler().fit_transform(feats)

        k_max = min(10, len(rfm) - 1)
        scores = evaluate_k(X, k_max)
        best_sil = int(scores.loc[scores["Silhouette"].idxmax(), "k"])
        best_elbow = elbow_k(scores["k"].tolist(), scores["Inertia"].tolist())

        st.markdown("#### Choosing the number of clusters")
        a, b, c = st.columns(3)
        fig = px.line(scores, x="k", y="Inertia", markers=True, title="Elbow method (lower is tighter)")
        fig.add_vline(x=best_elbow, line_dash="dash", line_color="#d62728",
                      annotation_text=f"elbow k={best_elbow}")
        with a:
            show(fig, 330)
        fig = px.line(scores, x="k", y="Silhouette", markers=True, title="Silhouette score (higher is better)")
        fig.add_vline(x=best_sil, line_dash="dash", line_color="#2ca02c",
                      annotation_text=f"best k={best_sil}")
        with b:
            show(fig, 330)
        fig = px.line(scores, x="k", y="Davies-Bouldin", markers=True,
                      title="Davies-Bouldin index (lower is better)")
        with c:
            show(fig, 330)

        insight(f"<b>Recommendation:</b> silhouette peaks at <b>k = {best_sil}</b> and the elbow "
                f"is near <b>k = {best_elbow}</b>. When they disagree, pick the k whose clusters you "
                "can describe and act on.")
        with st.expander("All validation scores"):
            table(scores.round(3))

        k = st.slider("Number of clusters (k)", 2, k_max, best_sil)
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        rfm["cluster"] = km.labels_.astype(str)
        z = pd.DataFrame(X, columns=["recency", "frequency", "monetary"])
        z["cluster"] = km.labels_.astype(str)
        profile_z = z.groupby("cluster").mean()
        names = label_clusters(profile_z)
        rfm["segment"] = rfm["cluster"].map(names)

        st.markdown("#### Cluster results")
        left, right = st.columns(2)
        with left:
            pca = PCA(n_components=2, random_state=42)
            pts = pca.fit_transform(X)
            pca_df = pd.DataFrame({"PC1": pts[:, 0], "PC2": pts[:, 1],
                                   "cluster": rfm["cluster"], "customer_id": rfm["customer_id"]})
            ev = pca.explained_variance_ratio_.sum() * 100
            show(px.scatter(pca_df, x="PC1", y="PC2", color="cluster", hover_data=["customer_id"],
                            title=f"Clusters in 2D (PCA, {ev:.0f}% variance explained)"), 420)
        with right:
            show(px.scatter_3d(rfm, x="recency", y="frequency", z="monetary", color="cluster",
                               hover_data=["customer_id"], title="Clusters in RFM space"), 420)

        left, right = st.columns(2)
        with left:
            show(px.imshow(profile_z.round(2), text_auto=True, aspect="auto",
                           color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                           title="Cluster profile (standardized means)",
                           labels=dict(x="", y="Cluster", color="z-score")), 360)
        with right:
            sizes = rfm.groupby("cluster").agg(Customers=("customer_id", "count"),
                                               Revenue=("monetary", "sum")).reset_index()
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]],
                                subplot_titles=("Customers", "Revenue"))
            fig.add_trace(go.Pie(labels=sizes["cluster"], values=sizes["Customers"], hole=.45), 1, 1)
            fig.add_trace(go.Pie(labels=sizes["cluster"], values=sizes["Revenue"], hole=.45), 1, 2)
            fig.update_layout(title="Cluster size vs revenue share")
            show(fig, 360)

        summary = rfm.groupby(["cluster", "segment"]).agg(
            Customers=("customer_id", "nunique"),
            Avg_Recency=("recency", "mean"),
            Avg_Frequency=("frequency", "mean"),
            Avg_Spending=("monetary", "mean"),
            Total_Revenue=("monetary", "sum"),
        ).round(2).reset_index()
        table(summary)

# ----------------------------------------------------------------------------
# Customer classification (Decision tree)
# ----------------------------------------------------------------------------
with predict_tab:
    st.subheader("Customer Value Classification (Decision Tree)")
    st.caption("Classify customers as High or Low value from behaviour. Total spend and average order value "
               "are deliberately NOT used as inputs because they would reveal the answer.")

    latest = filtered["order_date"].max().normalize() + pd.Timedelta(days=1)
    cust = filtered.groupby("customer_id").agg(
        order_frequency=("order_id", "nunique"),
        total_quantity=("quantity", "sum"),
        total_spending=("sales_amount", "sum"),
        distinct_items=("item_id", "nunique"),
        distinct_categories=("category", "nunique"),
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
    ).reset_index()
    cust["avg_items_per_order"] = cust["total_quantity"] / cust["order_frequency"].clip(lower=1)
    cust["recency_days"] = (latest - cust["last_order"]).dt.days
    cust["lifetime_days"] = (cust["last_order"] - cust["first_order"]).dt.days
    feature_cols = ["order_frequency", "total_quantity", "avg_items_per_order",
                    "distinct_items", "distinct_categories", "recency_days", "lifetime_days"]

    p1, p2, p3, p4 = st.columns(4)
    top_pct = p1.slider("High value = top X% of spenders", 10, 50, 50, 5)
    max_depth = p2.slider("Max tree depth", 1, 10, 4)
    test_size = p3.slider("Test size", 0.1, 0.4, 0.25, 0.05)
    criterion = p4.selectbox("Split criterion", ["gini", "entropy"])

    threshold = cust["total_spending"].quantile(1 - top_pct / 100)
    cust["value_class"] = np.where(cust["total_spending"] >= threshold, "High Value", "Low Value")
    y = (cust["value_class"] == "High Value").astype(int)

    if len(cust) < 20 or y.value_counts().min() < 5:
        st.warning("Not enough customers (or too few in one class) to train a reliable model. "
                   "Widen the filters or change the high-value percentage.")
    else:
        X = cust[feature_cols]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                       class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, proba)
        cv_folds = int(min(5, y_train.value_counts().min()))
        cv_scores = cross_val_score(
            DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                   class_weight="balanced", random_state=42),
            X_train, y_train, cv=cv_folds)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Test accuracy", f"{accuracy_score(y_test, pred) * 100:.1f}%")
        m2.metric("Precision", f"{precision_score(y_test, pred, zero_division=0) * 100:.1f}%")
        m3.metric("Recall", f"{recall_score(y_test, pred, zero_division=0) * 100:.1f}%")
        m4.metric("F1 score", f"{f1_score(y_test, pred, zero_division=0) * 100:.1f}%")
        m5.metric("ROC AUC", f"{auc(fpr, tpr):.3f}")
        st.caption(f"{len(cust):,} customers • threshold: spend ≥ {money(threshold)} • "
                   f"{cv_folds}-fold CV accuracy on training data: {cv_scores.mean() * 100:.1f}% ± {cv_scores.std() * 100:.1f}%")

        left, right = st.columns(2)
        with left:
            cm = confusion_matrix(y_test, pred, labels=[0, 1])
            show(px.imshow(cm, text_auto=True, color_continuous_scale="Oranges",
                           x=["Pred: Low", "Pred: High"], y=["Actual: Low", "Actual: High"],
                           title="Confusion matrix (test set)"), 380)
        with right:
            fig = go.Figure()
            fig.add_scatter(x=fpr, y=tpr, name=f"Decision tree (AUC {auc(fpr, tpr):.2f})", mode="lines",
                            line=dict(width=3))
            fig.add_scatter(x=[0, 1], y=[0, 1], name="Random guess", line=dict(dash="dash", color="grey"))
            fig.update_layout(title="ROC curve", xaxis_title="False positive rate",
                              yaxis_title="True positive rate", legend=dict(orientation="h", y=-.2))
            show(fig, 380)

        left, right = st.columns(2)
        with left:
            imp = (pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
                   .sort_values("importance"))
            show(px.bar(imp, x="importance", y="feature", orientation="h",
                        title="Feature importance"), 380)
        with right:
            depth_rows = []
            for d in range(1, 13):
                t = DecisionTreeClassifier(max_depth=d, criterion=criterion,
                                           class_weight="balanced", random_state=42).fit(X_train, y_train)
                depth_rows.append({"Depth": d, "Train": t.score(X_train, y_train),
                                   "Test": t.score(X_test, y_test)})
            depth_df = pd.DataFrame(depth_rows).melt("Depth", var_name="Set", value_name="Accuracy")
            fig = px.line(depth_df, x="Depth", y="Accuracy", color="Set", markers=True,
                          title="Accuracy vs tree depth (overfitting check)")
            fig.add_vline(x=max_depth, line_dash="dash", annotation_text="selected")
            show(fig, 380)

        st.markdown("#### Decision tree")
        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(max(12, 3.2 * max_depth), 2.2 + 1.8 * max_depth))
            plot_tree(model, feature_names=feature_cols, class_names=["Low Value", "High Value"],
                      filled=True, rounded=True, fontsize=8, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Install matplotlib (`pip install matplotlib`) to draw the tree. Showing text rules instead.")
        with st.expander("Tree rules as text"):
            st.code(export_text(model, feature_names=feature_cols))

        with st.expander("Test-set predictions"):
            out = cust.loc[X_test.index, ["customer_id", "total_spending", "value_class"]].copy()
            out["predicted"] = np.where(pred == 1, "High Value", "Low Value")
            out["P(high value)"] = proba.round(3)
            table(out)

# ----------------------------------------------------------------------------
# Market basket (Apriori)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Running Apriori…")
def run_apriori(pairs, top_n, min_support, max_len):
    all_orders = pairs["order_id"].unique()
    top_names = pairs["name"].value_counts().head(top_n).index
    sub = pairs[pairs["name"].isin(top_names)].drop_duplicates()
    baskets = pd.crosstab(sub["order_id"], sub["name"]).gt(0)
    baskets = baskets.reindex(all_orders, fill_value=False)  # support is measured over ALL orders
    frequent = apriori(baskets, min_support=min_support, use_colnames=True, max_len=max_len)
    return frequent, len(baskets)


def build_rules(frequent, n_orders, min_conf):
    try:
        return association_rules(frequent, num_itemsets=n_orders, metric="confidence",
                                 min_threshold=min_conf)
    except TypeError:  # older mlxtend versions
        return association_rules(frequent, metric="confidence", min_threshold=min_conf)


with basket_tab:
    st.subheader("Market Basket Analysis (Apriori)")
    pairs = filtered[["order_id", "name"]]

    if pairs["order_id"].nunique() < 2 or pairs["name"].nunique() < 2:
        st.warning("Not enough orders or menu items for basket analysis.")
    else:
        b1, b2, b3, b4, b5 = st.columns(5)
        top_n = b1.slider("Top N items", 10, 100, 40, 5, help="Limits memory use on large menus")
        min_support = b2.slider("Min support", 0.005, 0.10, 0.01, 0.005)
        min_confidence = b3.slider("Min confidence", 0.10, 1.00, 0.20, 0.05)
        min_lift = b4.slider("Min lift", 1.0, 5.0, 1.0, 0.1)
        max_len = b5.slider("Max itemset size", 2, 4, 3)

        try:
            frequent, n_orders = run_apriori(pairs, top_n, min_support, max_len)
            if frequent.empty:
                st.info("No frequent itemsets found. Lower the minimum support.")
            else:
                frequent = frequent.copy()
                frequent["size"] = frequent["itemsets"].apply(len)
                frequent["itemset"] = frequent["itemsets"].apply(lambda s: " + ".join(sorted(map(str, s))))
                frequent["orders"] = (frequent["support"] * n_orders).round().astype(int)

                st.markdown("### 1. Frequent itemsets")
                sizes = sorted(frequent["size"].unique())
                chosen_sizes = st.multiselect("Itemset sizes to show", sizes, default=sizes)
                fi = (frequent[frequent["size"].isin(chosen_sizes)]
                      .sort_values("support", ascending=False))
                left, right = st.columns([3, 2])
                with left:
                    topfi = fi.head(15).sort_values("support")
                    fig = px.bar(topfi, x="support", y="itemset", orientation="h",
                                 color=topfi["size"].astype(str), title="Top frequent itemsets",
                                 labels={"color": "Items in set"})
                    show(fig, 480)
                with right:
                    counts = frequent.groupby("size").size().reset_index(name="Itemsets")
                    show(px.bar(counts, x="size", y="Itemsets", title="Number of frequent itemsets by size",
                                labels={"size": "Itemset size"}), 480)
                table(fi[["itemset", "size", "support", "orders"]].head(50),
                      column_config={"support": st.column_config.ProgressColumn(
                          "Support", min_value=0, max_value=float(fi["support"].max()), format="%.3f")})

                st.markdown("### 2. Association rules")
                rules = build_rules(frequent[["support", "itemsets"]], n_orders, min_confidence)
                rules = rules[rules["lift"] >= min_lift]
                if rules.empty:
                    st.info("No association rules for these settings. Lower confidence, lift or support.")
                else:
                    rules = rules.copy()
                    rules["single"] = (rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)
                    for col in ("antecedents", "consequents"):
                        rules[col] = rules[col].apply(lambda s: ", ".join(sorted(map(str, s))))
                    rules = rules.sort_values(["lift", "confidence"], ascending=False)

                    left, right = st.columns(2)
                    with left:
                        show(px.scatter(rules.head(300), x="support", y="confidence", color="lift",
                                        size="lift", hover_data=["antecedents", "consequents"],
                                        color_continuous_scale="Viridis",
                                        title="Rules: support vs confidence (colour/size = lift)"), 420)
                    with right:
                        single = rules[rules["single"]].head(40)
                        top_a = single["antecedents"].value_counts().head(12).index
                        top_c = single["consequents"].value_counts().head(12).index
                        heat = (single[single["antecedents"].isin(top_a) & single["consequents"].isin(top_c)]
                                .pivot_table(index="antecedents", columns="consequents", values="lift"))
                        if heat.shape[0] >= 2 and heat.shape[1] >= 2:
                            show(px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                                           title="Lift heatmap (single-item rules)",
                                           labels=dict(x="Then buys", y="If buys", color="Lift")), 420)
                        else:
                            show(px.histogram(rules, x="lift", nbins=25, title="Distribution of lift"), 420)

                    table(rules[["antecedents", "consequents", "support", "confidence", "lift", "leverage"]]
                          .head(30).round(3))

                    st.markdown("### 3. Analysis & recommendations")
                    basket_sizes = filtered.groupby("order_id")["name"].nunique()
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Orders analysed", f"{n_orders:,}")
                    k2.metric("Avg distinct items / order", f"{basket_sizes.mean():.2f}")
                    k3.metric("Multi-item orders", f"{(basket_sizes > 1).mean() * 100:.1f}%")
                    k4.metric("Rules found", f"{len(rules):,}")

                    for _, r in rules.head(5).iterrows():
                        insight(f"Customers who order <b>{r['antecedents']}</b> also order "
                                f"<b>{r['consequents']}</b> in {r['confidence'] * 100:.0f}% of cases, "
                                f"{r['lift']:.2f}× more often than chance "
                                f"(support {r['support'] * 100:.1f}% of orders).")
                    best_conf = rules.sort_values("confidence", ascending=False).iloc[0]
                    insight(f"<b>Most reliable rule:</b> {best_conf['antecedents']} → {best_conf['consequents']} "
                            f"(confidence {best_conf['confidence'] * 100:.0f}%, lift {best_conf['lift']:.2f}).")
                    st.markdown(
                        "**How to use this:** build combo deals from the top-lift pairs, place high-confidence "
                        "consequents as server or menu-card suggestions when the antecedent is ordered, and "
                        "check low-support rules before acting on them.")
                    st.caption("Support = share of orders containing the itemset; confidence = P(consequent | antecedent); "
                               "lift > 1 means the items are bought together more than chance. "
                               "Association is co-occurrence, not causation.")
        except Exception as e:
            st.warning(f"Basket analysis could not be completed: {e}")

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
with data_tab:
    st.subheader("Processed Dataset")
    st.write(f"Rows: {len(filtered):,} | Columns: {len(filtered.columns)}")
    table(filtered.head(500))
    with st.expander("Numeric summary"):
        st.dataframe(filtered[["quantity", "price", "sales_amount", "order_total"]].describe().round(2),
                     **STRETCH)
    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="restaurant_filtered_data.csv",
        mime="text/csv",
    )
