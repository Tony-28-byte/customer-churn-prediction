# Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to churn and estimates their churn probability.

The project includes an interactive Streamlit web application where users can enter customer information and receive a churn prediction with a risk level.

## Project Overview

Customer churn is an important business problem for telecom companies. Identifying customers who are likely to leave can help businesses take proactive retention actions.

This project uses a Random Forest classifier to predict customer churn based on customer demographic information, services, contract details, and payment method.

## Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains customer information such as:

- Demographics
- Tenure
- Monthly Charges
- Total Charges
- Phone and Internet services
- Online security and backup services
- Contract type
- Paperless billing
- Payment method

## Machine Learning Pipeline

The project uses a Scikit-learn Pipeline containing:

1. Data preprocessing using `ColumnTransformer`
2. Numerical feature handling
3. Categorical feature encoding using `OneHotEncoder`
4. Random Forest classification

### Model

The final model is a Random Forest Classifier with the following parameters:

```text
n_estimators = 200
max_depth = 10
min_samples_split = 5
min_samples_leaf = 2
random_state = 42
```

## Decision Threshold

Instead of using the default classification threshold of 0.50, a threshold of **0.30** was selected.

The lower threshold increases the model's ability to identify customers who may churn, which is useful in a customer-retention scenario where missing a potential churner can be costly.

The application therefore classifies a customer as likely to churn when:

```text
Churn Probability >= 0.30
```

Otherwise, the customer is classified as low churn risk.

## Model Evaluation

The final model was evaluated using a held-out test set.

At the selected threshold of 0.30:

- Churn Precision: 0.54
- Churn Recall: 0.78
- Churn F1-score: 0.64

The threshold was selected to prioritize recall for the churn class while maintaining a reasonable level of precision.

## Streamlit Application

The project includes an interactive Streamlit application.

The application allows users to enter customer information and provides:

- Churn probability
- Churn prediction
- Risk classification
- Decision based on the selected threshold

Example output:

```text
Churn Probability: 79%

High Churn Risk
```

Another example:

```text
Churn Probability: 1.38%

Low Churn Risk
```

## Project Structure

```text
Customer Churn Prediction/
│
├── data/
│   └── Telco-Customer-Churn/
│
├── app.py
├── customer_churn_model.pkl
├── churn_threshold.pkl
├── requirements.txt
├── Customer_Churn.ipynb
├── Telco_Customer_Churn.ipynb
└── README.md
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd Customer-Churn-Prediction
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest
- Joblib
- Streamlit
- Jupyter Notebook

## Key Skills Demonstrated

- Data preprocessing
- Exploratory Data Analysis
- Feature engineering
- Categorical encoding
- Machine Learning model training
- Random Forest classification
- Model evaluation
- Precision / Recall / F1-score analysis
- Decision threshold optimization
- Model serialization
- Streamlit application development
- ML project deployment

## Future Improvements

Possible future improvements include:

- Hyperparameter optimization
- Model comparison with XGBoost or Gradient Boosting
- Probability calibration
- Explainable AI using SHAP
- Customer segmentation
- Deployment with a public URL
- Monitoring model performance over time

## Author

Machine Learning project developed as part of a practical portfolio project focused on customer churn prediction.