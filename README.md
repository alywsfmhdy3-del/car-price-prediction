# 🚗 Car Price Prediction using Machine Learning

An end-to-end Machine Learning project that compares **7 different regression models** to predict car prices based on features like manufacturer, model, fuel type, and car age.

---

## 📌 Project Overview

The goal of this project is to build and compare multiple Machine Learning models to accurately predict the **price of a car** based on its specifications. This helps:
- **Buyers:** Estimate fair market value before purchasing.
- **Sellers:** Set competitive prices based on features.
- **Dealerships:** Automate pricing decisions.

The project uses a **scikit-learn Pipeline** to ensure clean preprocessing and prevent data leakage during cross-validation.

---

## 📂 Dataset

The dataset (`car_sales_data.csv`) contains information about cars and their selling prices.

| Feature | Description |
|---------|-------------|
| `Manufacturer` | Car brand (e.g., Toyota, BMW) |
| `Model` | Car model name |
| `Fuel type` | Petrol, Diesel, Electric, etc. |
| `Year of manufacture` | Manufacturing year (used to compute `age`) |
| `Price` | **Target Variable** |

> 🔧 **Feature Engineering:** Created a new feature `age = 2025 - Year of manufacture` to capture depreciation, which is a key factor in car pricing.

---

## 🛠️ Project Workflow

### 1. Import Libraries
Loaded essential libraries for data manipulation, visualization, modeling, and evaluation.

### 2. Load Data
Read the dataset from CSV into a Pandas DataFrame.

### 3. Data Preprocessing
- **Feature Engineering:** Created `age` from the manufacturing year.
- **Dropped** the original year column.
- **Handled missing values** by dropping rows with NaNs.
- **Separated** features (`X`) and target (`y`).

### 4. Preprocessing Pipeline
Used `ColumnTransformer` to apply different transformations:
- **Numerical columns** → `StandardScaler` (mean=0, std=1).
- **Categorical columns** (`Manufacturer`, `Fuel type`, `Model`) → `OneHotEncoder`.

### 5. Train-Test Split
- **80% Training** / **20% Testing**
- `random_state=42` for reproducibility.

### 6. Models Compared

| Category | Models |
|----------|--------|
| 📈 **Linear Models** | Linear Regression, Ridge, Lasso |
| 🌳 **Tree Models** | Decision Tree, Random Forest |
| 🚀 **Boosting** | Gradient Boosting, XGBoost |

### 7. Model Training & Evaluation
Each model was trained using a **Pipeline** and evaluated on the test set. Additionally, **5-fold Cross-Validation** was performed to ensure the results are robust.

Metrics used:
- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **R² Score** — Coefficient of Determination
- **CV R² Mean** — Cross-Validation R² (5-fold)

### 8. Visualization
Plotted **Actual vs Predicted** prices for the best-performing model to visually assess accuracy.

### 9. Hyperparameter Tuning
Applied **GridSearchCV** on Random Forest to find the best combination of:
- `n_estimators`: [100, 200]
- `max_depth`: [None, 10, 20]

### 10. Save the Model
Saved the best estimator using `joblib` for future use.

### 11. Load Model Example
Demonstrated how to load the saved model and make predictions on new samples.

---

## 🧰 Technologies Used

- **Python 3.x**
- **Pandas** & **NumPy** — Data manipulation
- **Matplotlib** & **Seaborn** — Visualization
- **Scikit-learn** — Preprocessing, Pipelines, Models, Metrics
- **XGBoost** — Gradient Boosting
- **Joblib** — Model persistence

---

## 📁 Project Structure
├── car_price_prediction.ipynb # Main notebook
├── car_sales_data.csv # Dataset
├── best_car_price_model.pkl # Saved trained model
├── README.md # Project documentation
└── requirements.txt # Dependencies

text

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/car-price-prediction.git
cd car-price-prediction
Install dependencies:

bash
pip install -r requirements.txt
Open the notebook:

bash
jupyter notebook car_price_prediction.ipynb
Run all cells to reproduce the results.

📊 Results
The project outputs a comparison table showing MAE, MSE, R², and CV R² for all 7 models. The best-performing model is then selected for hyperparameter tuning and saved to disk.

💡 Gradient Boosting models (XGBoost, Gradient Boosting, Random Forest) typically outperform linear models on this type of tabular data.

💡 Future Improvements
□ Add more advanced feature engineering (e.g., brand prestige score, mileage).
□ Use Optuna for smarter hyperparameter tuning.
□ Deploy the model using Streamlit or FastAPI for real-time predictions.
□ Add SHAP values for model explainability.
👤 Author
Youssef Mahdy

GitHub: alywsfmhdy3-del

Kaggle: @youssefmahdyabosaleh