# 🍽️ Restaurant Sales & Customer Behavior Analysis

A comprehensive Data Warehousing & Data Mining (DWDM) project and interactive analytics dashboard designed to analyze restaurant sales transactions, evaluate dining patterns, segment customers, predict purchasing behavior, and uncover market basket associations.

---

## 📌 Project Overview

This repository provides an end-to-end data intelligence solution for restaurant operations:
- **Data Warehousing & OLAP**: Schema design, data integration from relational CSV tables, ETL pipeline, and multidimensional OLAP operations (Roll-up, Drill-down, Slice, and Dice).
- **Machine Learning & Data Mining**:
  - **Customer Segmentation**: Unsupervised clustering using **K-Means** with optimal cluster validation (Silhouette analysis, Davies-Bouldin, Calinski-Harabasz) and **PCA** dimensionality reduction.
  - **Customer Classification**: Predictive modeling using **Decision Tree Classifiers** to classify customer spending tiers, evaluated via Confusion Matrix, ROC-AUC, Precision, Recall, and F1-score.
  - **Market Basket Analysis**: Association rule mining using the **Apriori Algorithm** to identify frequent menu item combinations, cross-selling opportunities, and rule metrics (Support, Confidence, Lift).
- **Interactive Web Dashboard**: Built with **Streamlit** and **Plotly** to provide business managers with real-time exploratory visualizations, filters, and actionable insights.

---

## 🗂️ Repository Structure

```plaintext
.
├── CODE/
│   ├── Restaurant_DWDM_Project.ipynb  # End-to-end DWDM notebook (SQL, ETL, OLAP, ML)
│   └── restaurant_dashboard.py        # Streamlit interactive multi-tab dashboard
├── DATASET/
│   ├── restaurant-sample-database-v1-csv-customers.csv
│   ├── restaurant-sample-database-v1-csv-food_and_beverage_catalog.csv
│   ├── restaurant-sample-database-v1-csv-food_category.csv
│   ├── restaurant-sample-database-v1-csv-orders.csv
│   ├── restaurant-sample-database-v1-csv-order_items.csv
│   └── restaurant-sample-database-v1-csv-payments.csv
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Details

The project utilizes relational restaurant transaction data structured across six tables:

| Dataset File | Description | Key Attributes |
|---|---|---|
| `customers.csv` | Customer demographic records | `customer_id`, `name`, `email`, `phone` |
| `food_and_beverage_catalog.csv` | Menu catalog & pricing | `item_id`, `item_name`, `category_id`, `price` |
| `food_category.csv` | Food and beverage classifications | `category_id`, `category_name` |
| `orders.csv` | Order header information | `order_id`, `customer_id`, `order_date`, `table_number` |
| `order_items.csv` | Line items per order | `order_item_id`, `order_id`, `item_id`, `quantity` |
| `payments.csv` | Payment transactions | `payment_id`, `order_id`, `payment_method`, `amount` |

---

## 🚀 Key Modules & Capabilities

### 1. 📈 Sales Analytics
- Temporal trends: hourly order patterns, peak dining hours, weekday vs. weekend distribution.
- Revenue contribution by menu category and individual items.
- Average order value (AOV) and payment method breakdown (Cash, Card, Digital wallets).

### 2. 👥 Customer Segmentation (K-Means & PCA)
- Features: Total spend, visit frequency, average basket size, and item diversity.
- Clustering algorithms: K-Means with standardized scaling.
- Evaluation metrics: Elbow method (Inertia), Silhouette Score, Davies-Bouldin index.
- 2D/3D visualization using Principal Component Analysis (PCA).

### 3. 🎯 Customer Classification (Decision Tree)
- Categorizes customer behavior (e.g., High-Value vs. Regular patrons).
- Evaluates model performance: Accuracy, Precision, Recall, F1-Score, and ROC Curve.
- Decision tree visualization and feature importance analysis.

### 4. 🛒 Market Basket Analysis (Apriori)
- Identifies items frequently purchased together in single dining orders.
- Calculates Association Rules:
  $$\text{Support}(X \rightarrow Y) = P(X \cup Y)$$
  $$\text{Confidence}(X \rightarrow Y) = P(Y \mid X)$$
  $$\text{Lift}(X \rightarrow Y) = \frac{P(X \cup Y)}{P(X) \cdot P(Y)}$$
- Generates combo meal recommendations and menu engineering suggestions.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+ installed on your system.
- Git installed.

### 1. Clone the repository
```bash
git clone https://github.com/harish-rbuilds/RESTAURANT-SALES-CUSTOMER-BEHAVIOR-ANALYSIS.git
cd RESTAURANT-SALES-CUSTOMER-BEHAVIOR-ANALYSIS
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

### 3. Install required packages
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run

### Run the Interactive Dashboard
Launch the Streamlit web dashboard:
```bash
streamlit run CODE/restaurant_dashboard.py
```
Open your browser at `http://localhost:8501`.

### Run the DWDM Jupyter Notebook
To inspect the ETL pipeline, OLAP operations, and in-depth model training:
```bash
jupyter notebook CODE/Restaurant_DWDM_Project.ipynb
```

---

## 🧰 Tech Stack

- **Language:** Python
- **Web App / UI:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Visualizations:** Plotly Express & Graph Objects, Matplotlib
- **Machine Learning / Mining:** Scikit-learn, MLxtend
- **Data Warehousing & OLAP:** SQLite / In-memory SQL relational integration

---

## 👤 Author
- **Harish** ([@harish-rbuilds](https://github.com/harish-rbuilds))
