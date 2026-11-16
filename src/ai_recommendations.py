import ollama
import pandas as pd


def generate_recommendations(df):
    """
    Generate AI-powered business recommendations
    using actual sales data and a local Ollama model.
    """

    if df.empty:
        return "No sales data is available for analysis."

    # Prepare business performance summaries
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    total_orders = df["Order ID"].nunique()

    profit_margin = (
        (total_profit / total_sales) * 100
        if total_sales != 0 else 0
    )

    category_summary = (
        df.groupby("Category")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum")
        )
        .round(2)
        .to_dict("index")
    )

    region_summary = (
        df.groupby("Region")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum")
        )
        .round(2)
        .to_dict("index")
    )

    top_products = (
        df.groupby("Product Name")["Sales"]
        .sum()
        .nlargest(5)
        .round(2)
        .to_dict()
    )

    # Create the business data for the AI
    business_data = {
        "Total Sales": round(total_sales, 2),
        "Total Profit": round(total_profit, 2),
        "Total Orders": int(total_orders),
        "Profit Margin (%)": round(profit_margin, 2),
        "Category Performance": category_summary,
        "Regional Performance": region_summary,
        "Top 5 Products": top_products
    }

    prompt = f"""
    You are an experienced business intelligence analyst.

    Analyse the following actual retail business data:

    {business_data}

    Generate 5 useful and specific business recommendations.

    Requirements:
    1. Identify important sales and profit patterns.
    2. Highlight underperforming categories or regions.
    3. Identify opportunities to increase revenue.
    4. Suggest practical ways to improve profitability.
    5. Recommend specific business actions.
    6. Explain the data-based reason behind every recommendation.
    7. Do not invent numbers or unsupported facts.
    8. If the available data is insufficient to support a
       conclusion, clearly mention the limitation.

    Format the response using:
    - Clear headings
    - Numbered recommendations
    - Supporting data
    - Practical action points

    Keep the analysis professional and easy to understand.
    """

    try:
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional retail business "
                        "analyst. Give evidence-based insights."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={"temperature": 0.3}
        )

        return response["message"]["content"]

    except Exception as e:
        return (
            f"Unable to generate AI recommendations: {e}\n\n"
            "Please check whether Ollama is running and "
            "the llama3.2:3b model is installed."
        )