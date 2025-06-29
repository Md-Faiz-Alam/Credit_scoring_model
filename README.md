# 💳 Credit Scoring Model

A machine learning project to predict an individual's credit default risk using historical financial data.

## 🎯 Objective

To build a predictive model that classifies whether a person is likely to default on a loan, based on their financial history and profile.

## 🧠 Approach

- Implemented and compared classification algorithms:
  - Logistic Regression
  - Decision Tree
  - Random Forest (final chosen model)
- Developed a Streamlit web app for live predictions and user interaction.

## 🔍 Key Features

- ✅ **Feature Engineering**:
  - Log transformation of income
  - Credit usage flags
  - Debt-to-Income × Income interaction terms
- 📊 **Model Evaluation**:
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- 🧪 **Streamlit UI**:
  - Users can manually input financial details or choose from predefined sample profiles.
  - Displays real-time predictions and probability of default.

## 📁 Dataset Overview

Includes the following features:
- `credit_usage`: Credit utilization ratio (0 to 1)
- `age`: Applicant's age
- `debt_ratio`: Debt-to-Income ratio
- `monthly_income`: Income per month (in ₹)
- `open_credit`: Number of open credit accounts
- `times_late`: Number of times payment was late 90+ days
- `real_estate`: Real estate loan count
- `dependents`: Number of dependents

> Additional engineered features used internally for training:
- Log of monthly income
- High utilization flag
- Debt × Income interaction

## 📈 Results

- Best performing model: **Random Forest**
- Achieved strong performance across all key metrics
- Interactive web interface for demo and testing

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/credit-scoring-model.git
cd credit-scoring-model

### 2. Install dependencies
```bash
pip install -r requirements.txt

### 3. Launch the Streamlit app
```bash
streamlit run app.py

📸 Screenshot

Screenshot

🧾 License
This project is open-source under the MIT License.