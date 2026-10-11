# Boston House Price Prediction 🏡

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-brightgreen.svg)](https://flask.palletsprojects.com/)
[![Library](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning web application predicting median housing values using 13 socioeconomic and architectural attributes. The project encompasses exploratory data analysis (EDA), statistical validation, leakage-free feature scaling, model serialization, and interactive deployment via a Flask web application and RESTful API.

---

## 📌 Project Overview & Highlights

- **Linear Regression Baseline:** Trained an Ordinary Least Squares (OLS) regression model to predict housing prices in thousands of dollars ($1,000s).
- **Leak-Free Preprocessing:** Standardized continuous features using `StandardScaler`, strictly fit on training splits to prevent data leakage during inference and cross-validation.
- **Dual Interface Deployment:**
  - **Interactive Web App:** Form-based UI for real-time house valuation estimates.
  - **RESTful API:** Postman/cURL-ready JSON endpoint (`/predict_api`) for programmatic consumption.
- **Statistical Rigor:** Evaluated on a holdout test split and reinforced with 5-fold cross-validation.

---

## 📊 Dataset & Preprocessing

- **Source:** Boston Housing Dataset (`HousingData.csv`)
- **Total Records:** 506 raw rows $\rightarrow$ **394 complete rows** post-cleaning.
- **Handling Missing Data:** 112 rows contained null values across attributes (`CRIM`, `ZN`, `INDUS`, `CHAS`, etc.). For this baseline implementation, incomplete observations were dropped (`dropna()`). Future iterations can explore median or iterative KNN imputation.
- **Target Variable:** `MEDV` (renamed to `Price`), measuring median value of owner-occupied homes in **$1,000s**.

### Feature Summary (13 Predictors)
| Feature | Description |
| :--- | :--- |
| **CRIM** | Per capita crime rate by town |
| **ZN** | Proportion of residential land zoned for lots over 25,000 sq.ft. |
| **INDUS** | Proportion of non-retail business acres per town |
| **CHAS** | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
| **NOX** | Nitric oxides concentration (parts per 10 million) |
| **RM** | Average number of rooms per dwelling |
| **AGE** | Proportion of owner-occupied units built prior to 1940 |
| **DIS** | Weighted distances to five Boston employment centres |
| **RAD** | Index of accessibility to radial highways |
| **TAX** | Full-value property-tax rate per $10,000 |
| **PTRATIO**| Pupil-teacher ratio by town |
| **B** | Proportion of Black residents by town ($1000(Bk - 0.63)^2$) |
| **LSTAT** | Percentage of lower status of the population |

---

## 📈 Model Performance & Validation

Features were scaled and evaluated using a **70/30 train-test split** (275 train / 119 test observations), with generalizability verified across **5-fold cross-validation**:

| Metric | Holdout Test Set (70/30 Split) | 5-Fold Cross Validation |
| :--- | :---: | :---: |
| **R² Score** | **0.6905** | **0.7330 ± 0.0587** |
| **Adjusted R²** | **0.6522** | — |
| **Mean Absolute Error (MAE)** | **$3.46k** ($3,456) | **$3.28k ± 0.16k** ($3,280) |
| **Root Mean Squared Error (RMSE)** | **$5.37k** ($5,373) | **$4.65k ± 0.54k** ($4,648) |

> **Interpretation:** The model achieves a mean absolute error of ~$3,456 on unseen test homes, with cross-validation confirming an average $R^2$ of ~0.73 across fold distributions.

---

## 🛠️ Repository Structure

```text
HousePricePrediction/
├── app.py                                   # Flask web server and API endpoints
├── regmodel.pkl                             # Serialized Linear Regression model
├── scaling.pkl                              # Serialized StandardScaler pipeline
├── HousingData.csv                          # Dataset
├── Linear Regression ML Implementation.ipynb# Complete EDA, training, and CV notebook
├── requirements.txt                         # Project dependencies
├── templates/
│   └── home.html                            # Frontend web interface
├── LICENSE                                  # MIT License
└── README.md                                # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/HousePricePrediction.git
cd HousePricePrediction
```

### 2. Create and Activate Virtual Environment
* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

---

## 🔌 API Endpoints & Usage

### Web Interface (`/` & `/predict`)
Navigate to `http://127.0.0.1:5000/`, populate the 13 feature inputs in the HTML form, and submit to view the predicted home valuation.

### Programmatic Prediction (`/predict_api`)
Send a `POST` request with JSON containing the 13 feature attributes:

#### Request Example (cURL):
```bash
curl -X POST http://127.0.0.1:5000/predict_api \
     -H "Content-Type: application/json" \
     -d '{
       "data": {
         "CRIM": 0.00632,
         "ZN": 18.0,
         "INDUS": 2.31,
         "CHAS": 0.0,
         "NOX": 0.538,
         "RM": 6.575,
         "AGE": 65.2,
         "DIS": 4.0900,
         "RAD": 1.0,
         "TAX": 296.0,
         "PTRATIO": 15.3,
         "B": 396.90,
         "LSTAT": 4.98
       }
     }'
```

#### Response Example:
```json
{
  "Prediction": 29.48758966953259
}
```
*(Represents a predicted price of ~$29,488 USD)*.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
