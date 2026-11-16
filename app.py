
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
from src.ai_recommendations import generate_recommendations

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Sales Analytics",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cleaned_superstore.csv"
MODEL_PATH = BASE_DIR / "models" / "sales_forecasting_model.pkl"

# ==========================================
# 2. PREMIUM CSS THEME
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #F7F7F5;
    color: #292929;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3 {
    color: #B28A45;
    font-family: Georgia, serif;
}

section[data-testid="stSidebar"] {
    background-color: #252525;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #F5F5F0 !important;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E8E2D5;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
    min-height: 115px;
}

div[data-testid="stMetricLabel"] {
    color: #777777 !important;
}

div[data-testid="stMetricValue"] {
    color: #292929 !important;
    font-weight: 700;
    font-size: clamp(17px, 1.6vw, 26px) !important;
    overflow-wrap: anywhere;
}

.stDownloadButton button {
    background-color: #B28A45;
    color: white;
    border-radius: 8px;
    border: none;
}

hr {
    border-color: #D6C6A5 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    required = [
        "Order Date", "Category", "Region",
        "Sales", "Profit", "Order ID", "Product Name"
    ]

    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data["Order Date"] = pd.to_datetime(
        data["Order Date"], errors="coerce"
    )

    for col in ["Sales", "Profit", "Quantity", "Discount"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data.dropna(
        subset=["Order Date", "Sales", "Profit"],
        inplace=True
    )

    return data


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

if df.empty:
    st.error("The dataset contains no valid records.")
    st.stop()

# ==========================================
# 4. PREMIUM DASHBOARD HEADER
# ==========================================

st.html("""
<div style="
    background: linear-gradient(120deg, #252525, #454039);
    padding: 40px;
    border-radius: 14px;
    border-left: 6px solid #B28A45;
    margin-bottom: 30px;
">
    <p style="
        color: #D6B875;
        font-size: 13px;
        letter-spacing: 3px;
        font-weight: bold;
        margin-bottom: 12px;
    ">BUSINESS INTELLIGENCE | SALES ANALYTICS</p>

    <h1 style="
        color: #FFFFFF;
        font-family: Georgia, serif;
        font-size: 38px;
        margin: 0;
    ">Smart Sales Analytics</h1>

    <p style="
        color: #E1DED7;
        font-size: 16px;
        margin-top: 15px;
        margin-bottom: 0;
    ">Transform historical sales data into meaningful business insights and intelligent forecasts.</p>
</div>
""")

# ==========================================
# 5. SIDEBAR FILTERS
# ==========================================

st.sidebar.title("Dashboard Filters")
st.sidebar.caption("Refine the sales data shown in the dashboard.")

categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())
regions = ["All"] + sorted(df["Region"].dropna().unique().tolist())

selected_category = st.sidebar.selectbox(
    "Select Category", categories
)

selected_region = st.sidebar.selectbox(
    "Select Region", regions
)

min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()

selected_dates = st.sidebar.date_input(
    "Select Order Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

filtered_df = df.copy()

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["Category"] == selected_category
    ]

if selected_region != "All":
    filtered_df = filtered_df[
        filtered_df["Region"] == selected_region
    ]

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered_df = filtered_df[
        (filtered_df["Order Date"].dt.date >= start_date) &
        (filtered_df["Order Date"].dt.date <= end_date)
    ]

st.sidebar.caption(f"Showing {len(filtered_df):,} sales records")

if filtered_df.empty:
    st.warning("No records found for the selected filters.")
    st.stop()

# ==========================================
# 6. BUSINESS PERFORMANCE OVERVIEW
# ==========================================

st.subheader("Business Performance Overview")

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order ID"].nunique()
average_sales = filtered_df["Sales"].mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric("💰 Total Sales", f"${total_sales:,.2f}")
c2.metric("📈 Total Profit", f"${total_profit:,.2f}")
c3.metric("📦 Total Orders", f"{total_orders:,}")
c4.metric("🛒 Average Sales", f"${average_sales:,.2f}")

st.divider()

# ==========================================
# 7. EXECUTIVE INSIGHTS
# ==========================================

st.subheader("Executive Insights")

category_summary = filtered_df.groupby("Category")["Sales"].sum()
region_summary = filtered_df.groupby("Region")["Sales"].sum()
profit_summary = filtered_df.groupby("Category")["Profit"].sum()

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Top Sales Category", category_summary.idxmax())
    st.caption(f"Revenue: ${category_summary.max():,.2f}")

with c2:
    st.metric("Leading Region", region_summary.idxmax())
    st.caption(f"Revenue: ${region_summary.max():,.2f}")

with c3:
    st.metric("Most Profitable Category", profit_summary.idxmax())
    st.caption(f"Profit: ${profit_summary.max():,.2f}")


# ==========================================
# ADDITIONAL BUSINESS PERFORMANCE KPIs
# ==========================================

st.subheader("Business Growth & Profitability")

monthly_kpi = (
    filtered_df.set_index("Order Date")
    .resample("ME")["Sales"]
    .sum()
)

if len(monthly_kpi) >= 2 and monthly_kpi.iloc[-2] != 0:
    sales_growth = (
        (monthly_kpi.iloc[-1] - monthly_kpi.iloc[-2])
        / monthly_kpi.iloc[-2]
    ) * 100
else:
    sales_growth = None

profit_margin = (
    (total_profit / total_sales) * 100
    if total_sales != 0 else 0
)

kpi1, kpi2 = st.columns(2)

with kpi1:
    if sales_growth is not None:
        st.metric(
            "📈 Monthly Sales Growth",
            f"{sales_growth:.2f}%",
            delta=f"{sales_growth:.2f}%"
        )
    else:
        st.metric("📈 Monthly Sales Growth", "Not available")

with kpi2:
    st.metric(
        "💹 Profit Margin",
        f"{profit_margin:.2f}%"
    )

st.divider()

# ==========================================
# 8. MONTHLY SALES TREND
# ==========================================

st.subheader("Monthly Sales Trend")

monthly_sales = (
    filtered_df.set_index("Order Date")
    .resample("ME")["Sales"]
    .sum()
    .reset_index()
)

fig = px.line(
    monthly_sales,
    x="Order Date",
    y="Sales",
    markers=True,
    title="Monthly Revenue Trend"
)

fig.update_traces(
    line=dict(color="#B28A45", width=3),
    marker=dict(color="#252525", size=7)
)

fig.update_layout(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",
    hovermode="x unified",
    xaxis_title="Month",
    yaxis_title="Revenue ($)"
)

st.plotly_chart(fig, width="stretch")

# ==========================================
# YEAR-OVER-YEAR SALES COMPARISON
# ==========================================

st.subheader("Year-over-Year Sales Comparison")

yearly_monthly = filtered_df.copy()

yearly_monthly["Year"] = yearly_monthly["Order Date"].dt.year
yearly_monthly["Month"] = yearly_monthly["Order Date"].dt.month_name()
yearly_monthly["Month Number"] = yearly_monthly["Order Date"].dt.month

comparison_data = (
    yearly_monthly.groupby(
        ["Year", "Month", "Month Number"]
    )["Sales"]
    .sum()
    .reset_index()
    .sort_values("Month Number")
)

fig_comparison = px.line(
    comparison_data,
    x="Month",
    y="Sales",
    color="Year",
    markers=True,
    title="Monthly Sales Performance Across Years",
    color_discrete_sequence=[
        "#B28A45", "#252525", "#8A9A8A", "#A67C7C"
    ]
)

fig_comparison.update_layout(
    template="plotly_white",
    xaxis_title="Month",
    yaxis_title="Sales Revenue ($)",
    hovermode="x unified",
    legend_title="Year"
)

st.plotly_chart(fig_comparison, width="stretch")


# ==========================================
# 9. CATEGORY AND REGION ANALYSIS
# ==========================================

st.subheader("Sales Performance Analysis")

left, right = st.columns(2)

with left:
    category_sales = (
        filtered_df.groupby("Category")["Sales"]
        .sum()
        .reset_index()
        .sort_values("Sales", ascending=False)
    )

    fig_category = px.bar(
        category_sales,
        x="Category",
        y="Sales",
        title="Sales by Product Category",
        color_discrete_sequence=["#B28A45"]
    )

    fig_category.update_layout(
        template="plotly_white",
        showlegend=False
    )

    st.plotly_chart(fig_category, use_container_width=True)

with right:
    region_sales = (
        filtered_df.groupby("Region")["Sales"]
        .sum()
        .reset_index()
    )

    fig_region = px.pie(
        region_sales,
        names="Region",
        values="Sales",
        hole=0.5,
        title="Regional Sales Distribution",
        color_discrete_sequence=[
            "#252525", "#B28A45", "#D6C6A5", "#777777"
        ]
    )

    fig_region.update_layout(template="plotly_white")

    st.plotly_chart(fig_region, use_container_width=True)

# ==========================================
# 10. PROFIT ANALYSIS
# ==========================================

st.subheader("Profit Analysis")

profit_data = (
    filtered_df.groupby("Category")["Profit"]
    .sum()
    .reset_index()
)

fig_profit = px.bar(
    profit_data,
    x="Category",
    y="Profit",
    title="Profit by Product Category",
    color_discrete_sequence=["#927039"]
)

fig_profit.update_layout(
    template="plotly_white",
    showlegend=False
)

st.plotly_chart(fig_profit, use_container_width=True)

# ==========================================
# 11. TOP 10 PRODUCTS
# ==========================================

st.subheader("Top 10 Best-Selling Products")

top_products = (
    filtered_df.groupby("Product Name")["Sales"]
    .sum()
    .nlargest(10)
    .reset_index()
    .sort_values("Sales", ascending=True)
)

fig_products = px.bar(
    top_products,
    x="Sales",
    y="Product Name",
    orientation="h",
    title="Top Products by Revenue",
    color_discrete_sequence=["#B28A45"]
)

fig_products.update_layout(
    template="plotly_white",
    showlegend=False,
    yaxis_title=""
)

st.plotly_chart(fig_products, use_container_width=True)


# ==========================================
# TOP 10 CUSTOMERS ANALYSIS
# ==========================================

st.divider()
st.subheader("Top 10 Valuable Customers")

customer_summary = (
    filtered_df.groupby("Customer Name")
    .agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Total_Orders=("Order ID", "nunique")
    )
    .reset_index()
    .sort_values("Total_Sales", ascending=False)
    .head(10)
)

left, right = st.columns(2)

with left:
    fig_customers = px.bar(
        customer_summary.sort_values("Total_Sales"),
        x="Total_Sales",
        y="Customer Name",
        orientation="h",
        title="Top 10 Customers by Revenue",
        color_discrete_sequence=["#B28A45"]
    )

    fig_customers.update_layout(
        template="plotly_white",
        xaxis_title="Total Sales ($)",
        yaxis_title="",
        showlegend=False
    )

    st.plotly_chart(fig_customers, width="stretch")

with right:
    fig_customer_profit = px.bar(
        customer_summary.sort_values("Total_Profit"),
        x="Total_Profit",
        y="Customer Name",
        orientation="h",
        title="Profit Generated by Top Customers",
        color_discrete_sequence=["#252525"]
    )

    fig_customer_profit.update_layout(
        template="plotly_white",
        xaxis_title="Total Profit ($)",
        yaxis_title="",
        showlegend=False
    )

    st.plotly_chart(fig_customer_profit, width="stretch")

st.markdown("#### Customer Performance Details")

st.dataframe(
    customer_summary.rename(columns={
        "Customer Name": "Customer",
        "Total_Sales": "Revenue ($)",
        "Total_Profit": "Profit ($)",
        "Total_Orders": "Orders"
    }).style.format({
        "Revenue ($)": "${:,.2f}",
        "Profit ($)": "${:,.2f}"
    }),
    width="stretch",
    hide_index=True
)


# ==========================================
# SALES AND PROFITABILITY HEATMAP
# ==========================================

st.divider()
st.subheader("Regional Sales & Profitability Heatmap")

heatmap_data = (
    filtered_df.groupby(["Region", "Category"])
    .agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum")
    )
    .reset_index()
)

heatmap_metric = st.radio(
    "Select Metric",
    ["Sales", "Profit"],
    horizontal=True,
    key="heatmap_metric"
)

if heatmap_metric == "Sales":
    value_column = "Total_Sales"
    color_title = "Total Sales ($)"
else:
    value_column = "Total_Profit"
    color_title = "Total Profit ($)"

heatmap_pivot = heatmap_data.pivot(
    index="Region",
    columns="Category",
    values=value_column
)

fig_heatmap = px.imshow(
    heatmap_pivot,
    text_auto=".2s",
    aspect="auto",
    color_continuous_scale=[
        "#F5EBD5",
        "#D6B875",
        "#B28A45",
        "#454039",
        "#252525"
    ],
    title=f"{color_title} by Region and Category",
    labels={
        "x": "Product Category",
        "y": "Region",
        "color": color_title
    }
)

fig_heatmap.update_layout(
    template="plotly_white",
    paper_bgcolor="white",
    title_font=dict(
        color="#252525",
        family="Georgia",
        size=20
    )
)

st.plotly_chart(
    fig_heatmap,
    width="stretch"
)

# ==========================================
# AI-POWERED BUSINESS RECOMMENDATIONS
# ==========================================

st.divider()

st.subheader("🤖 AI Business Intelligence")
st.caption(
    "Powered by Llama 3.2 | Local AI-powered sales analysis"
)

with st.container(border=True):
    st.markdown("### Generate Business Insights")

    st.write(
        "Let AI analyse your current dashboard data "
        "and identify opportunities to improve performance."
    )

    if st.button(
        "✨ Generate AI Recommendations",
        type="primary",
        width="stretch"
    ):
        with st.spinner(
            "AI is analysing your sales data. Please wait..."
        ):
            ai_result = generate_recommendations(filtered_df)

        st.session_state["ai_recommendations"] = ai_result

    if "ai_recommendations" in st.session_state:
        st.divider()

        st.markdown("### 📊 AI Analysis & Recommendations")

        st.markdown(
            st.session_state["ai_recommendations"]
        )

# ==========================================
# 12. MACHINE LEARNING FORECAST
# ==========================================

st.divider()
st.subheader("Machine Learning Sales Forecast")

if model is None:
    st.warning(
        "Forecast model not found. Run src/forecasting.py first."
    )
else:
    forecast_period = st.selectbox(
        "Select Forecast Period",
        [3, 6, 12],
        format_func=lambda x: f"{x} Months"
    )

    history = (
        df.set_index("Order Date")["Sales"]
        .resample("ME")
        .sum()
        .sort_index()
    )

    if len(history) < 3:
        st.warning("Not enough historical data for forecasting.")
    else:
        predictions = []
        sales_history = history.tolist()
        last_date = history.index[-1]

        feature_columns = [
            "Month",
            "Year",
            "Quarter",
            "Lag_1",
            "Lag_2",
            "Lag_3",
            "Rolling_Mean_3"
        ]

        for _ in range(forecast_period):
            next_date = last_date + pd.offsets.MonthEnd(1)

            input_data = pd.DataFrame([{
                "Month": next_date.month,
                "Year": next_date.year,
                "Quarter": next_date.quarter,
                "Lag_1": sales_history[-1],
                "Lag_2": sales_history[-2],
                "Lag_3": sales_history[-3],
                "Rolling_Mean_3": np.mean(sales_history[-3:])
            }], columns=feature_columns)

            prediction = float(model.predict(input_data)[0])

            predictions.append({
                "Date": next_date,
                "Predicted Sales": prediction
            })

            sales_history.append(prediction)
            last_date = next_date

        forecast_df = pd.DataFrame(predictions)

        forecast_total = forecast_df["Predicted Sales"].sum()

        st.metric(
            f"Predicted Sales - Next {forecast_period} Months",
            f"${forecast_total:,.2f}"
        )

        history_df = history.tail(12).reset_index()
        history_df.columns = ["Date", "Sales"]

        forecast_fig = go.Figure()

        forecast_fig.add_trace(go.Scatter(
            x=history_df["Date"],
            y=history_df["Sales"],
            mode="lines+markers",
            name="Historical Sales",
            line=dict(color="#252525", width=3)
        ))

        forecast_fig.add_trace(go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Predicted Sales"],
            mode="lines+markers",
            name="Forecasted Sales",
            line=dict(color="#B28A45", width=3, dash="dash")
        ))

        forecast_fig.update_layout(
            template="plotly_white",
            title="Historical vs Future Sales Forecast",
            xaxis_title="Month",
            yaxis_title="Sales ($)",
            hovermode="x unified"
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Forecast CSV",
            forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="sales_forecast.csv",
            mime="text/csv"
        )

# ==========================================
# 13. DOWNLOAD DATA
# ==========================================

st.divider()
st.subheader("Download Sales Data")

st.download_button(
    "Download Filtered Sales Data",
    filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_sales_data.csv",
    mime="text/csv"
)

# ==========================================
# 14. FOOTER
# ==========================================

st.html("""
<div style="
    text-align: center;
    padding: 25px;
    border-top: 1px solid #D6C6A5;
    margin-top: 30px;
">
    <p style="color: #927039; font-weight: bold; letter-spacing: 2px;">
        SMART SALES ANALYTICS
    </p>
    <p style="color: #777777; font-size: 13px;">
        Built using Python | Pandas | Plotly | Scikit-learn | Streamlit
    </p>
</div>
""")

st.success("Smart Sales Analytics Dashboard loaded successfully!")
