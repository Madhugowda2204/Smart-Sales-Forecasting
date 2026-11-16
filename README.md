# 📊 Smart Sales Forecasting & AI-Powered Business Analytics

An intelligent sales analytics dashboard that combines Machine Learning, data visualization and local AI to analyse retail sales performance, forecast future revenue and generate actionable business recommendations.

## 🚀 Project Overview

The **Smart Sales Forecasting & Analytics Dashboard** helps businesses understand historical sales patterns, monitor performance and make data-driven decisions.

Built using Python and Streamlit, the application integrates a Random Forest forecasting model with a locally running Llama 3.2 model through Ollama to provide dynamic business insights.

## ✨ Key Features

### 1. Interactive Sales Analytics
- Total sales, profit, orders and average sales KPIs.
- Category-wise and region-wise sales analysis.
- Monthly sales trend visualization.
- Year-over-year sales comparison.
- Profit margin and monthly sales growth.

### 2. Machine Learning Sales Forecasting
- Random Forest Regressor for sales prediction.
- Historical sales trend analysis.
- 3-month, 6-month and 12-month forecasting options.
- Model evaluation using MAE, RMSE and R².

### 3. AI-Powered Business Recommendations
- Integrated local Llama 3.2 model using Ollama.
- Analyses actual sales and profit summaries.
- Identifies category and regional performance patterns.
- Generates dynamic business recommendations.
- Provides actionable suggestions for improving revenue and profitability.

### 4. Customer and Product Analytics
- Top 10 valuable customers.
- Top-performing products.
- Customer revenue and profit analysis.

### 5. Regional Performance Analysis
- Regional sales and profitability heatmap.
- Category-wise regional comparisons.

### 6. Interactive Dashboard
- Category, region and date filters.
- Interactive Plotly visualizations.
- Downloadable filtered sales data.

## 🛠️ Technologies Used

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, Random Forest |
| Data Visualization | Plotly |
| Dashboard | Streamlit |
| Local AI | Ollama, Llama 3.2 |
| Model Storage | Joblib |
| Version Control | Git, GitHub |

## 🧠 Machine Learning Model

The project uses a Random Forest Regressor to forecast sales.

**Model configuration:**
- Algorithm: Random Forest Regressor
- Number of estimators: 300
- Maximum depth: 8
- Random state: 42

**Input features:**
- Month
- Year
- Quarter
- Previous month sales (lag features)
- Three-month rolling average

**Model evaluation:**

| Metric | Result |
|---|---:|
| MAE | 14,474.45 |
| RMSE | 18,620.62 |
| R² Score | 0.4434 |

These are baseline evaluation results from the current model. Further tuning and validation can improve forecasting performance.

## 🤖 AI Integration

The application integrates Ollama to run the Llama 3.2 language model locally.

The AI recommendation engine receives aggregated business metrics and generates contextual insights based on the selected dashboard filters.

Unlike predefined recommendation rules, the language model generates its responses dynamically from the supplied business data.

## 📁 Project Structure

```text
Smart-Sales-Forecasting/
│
├── data/
│   ├── superstore.csv
│   ├── cleaned_superstore.csv
│   └── charts/
│
├── models/
│   └── sales_forecasting_model.pkl
│
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── forecasting.py
│   └── ai_recommendations.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- Ollama installed on your computer

### 1. Clone the repository

```bash
git clone https://github.com/Madhugowda2204/Smart-Sales-Forecasting.git
```

### 2. Navigate to the project directory

```bash
cd Smart-Sales-Forecasting
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the AI model

```bash
ollama pull llama3.2:3b
```

### 7. Start Ollama

Make sure the Ollama application is running.

### 8. Launch the dashboard

```bash
streamlit run app.py
```

Open the application at:

```text
http://localhost:8501
```

## 📈 Future Enhancements

- Improve forecasting accuracy through hyperparameter tuning.
- Add more advanced forecasting algorithms.
- Introduce automated anomaly detection.
- Add downloadable business intelligence reports.
- Explore deployment with a hosted AI inference service.

## 👩‍💻 Author

**Madhugowda V**

GitHub: [Madhugowda2204](https://github.com/Madhugowda2204)

---

⭐ If you find this project interesting, consider giving the repository a star!
