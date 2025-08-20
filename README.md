**Clinical Trial Outcome Prediction**

This project aims to predict the completion status of clinical trials using machine learning techniques. The workflow involves data preprocessing, feature engineering, model training with XGBoost, and deployment of an interactive web application using Streamlit.

**⚙️ Features**

Preprocessing Notebook:
Removes redundant columns (e.g., IDs, text-heavy fields).
Handles missing values with categorical replacement & median imputation.
Performs feature engineering (trial duration, time to completion, feature interactions).

XGBoost Model:
Trained on cleaned features to predict trial completion (Completed vs Not Completed).
Evaluated using metrics such as Accuracy, Precision, Recall, F1-score, and AUC-ROC.

Streamlit Web App:
User-friendly interface to input trial details (conditions, sponsor, study design, etc.).
Generates predictions with probabilities.
Provides explainability using LIME for model interpretation.

**📊 Example Use Case**

Input: Trial details such as condition, sponsor, design, enrollment, and phase.
Output: Predicted trial completion status (Completed / Not Completed) with probabilities.
Explainability: LIME highlights the most important features contributing to the prediction.

**🛠️ Tech Stack**

Languages: Python
Libraries: Pandas, NumPy, Scikit-learn, XGBoost, LIME, Streamlit
Visualization: Matplotlib, Seaborn (in notebooks)
