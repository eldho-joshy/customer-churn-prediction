import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_auc_score, PrecisionRecallDisplay)

# DATA LOADING & PREPROCESSING
file_path = r"C:\Customer_Churn_Prediction\Data\Telco_customer_churn.xlsx"
df = pd.read_excel(file_path)

# Data Cleaning: Handle the 'Total Charges' string-to-numeric conversion
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df.dropna(subset=['Total Charges'], inplace=True)

# Save a copy for EDA visualizations
df_viz = df.copy()

# Drop identifiers and leakage columns
cols_to_drop = ['CustomerID', 'Count', 'Country', 'State', 'Lat Long', 'Latitude', 'Longitude', 'Zip Code', 'Churn Reason', 'Churn Label']
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)


# BUSINESS INSIGHT DASHBOARD (EDA)
plt.style.use('ggplot')
fig, axes = plt.subplots(2, 2, figsize=(18, 15))
plt.subplots_adjust(hspace=0.6, wspace=0.4) # Generous spacing between plots

# Chart 1: Tenure Risk
sns.kdeplot(df_viz.loc[df_viz['Churn Value'] == 0, 'Tenure Months'], label='Stayed', fill=True, ax=axes[0,0], color='#2ecc71')
sns.kdeplot(df_viz.loc[df_viz['Churn Value'] == 1, 'Tenure Months'], label='Churned', fill=True, ax=axes[0,0], color='#e74c3c')
axes[0,0].set_title('Customer Loyalty: Tenure vs Churn Risk', fontsize=15, pad=25)
axes[0,0].set_xlabel('Months with Company', labelpad=15)
axes[0,0].legend()

# Chart 2: Contract Impact
sns.countplot(x='Contract', hue='Churn Label', data=df_viz, ax=axes[0,1], palette='viridis')
axes[0,1].set_title('Impact of Contract Type on Retention', fontsize=15, pad=25)
axes[0,1].set_xlabel('Contract Structure', labelpad=15)

# Chart 3: Pricing Sensitivity
sns.boxplot(x='Churn Label', y='Monthly Charges', data=df_viz, ax=axes[1,0], palette='Set2')
axes[1,0].set_title('Pricing Sensitivity Analysis', fontsize=15, pad=25)
axes[1,0].set_ylabel('Monthly Charges ($)', labelpad=15)

# Chart 4: Internet Service
sns.countplot(x='Internet Service', hue='Churn Label', data=df_viz, ax=axes[1,1], palette='magma')
axes[1,1].set_title('Churn Rate by Internet Service', fontsize=15, pad=25)
axes[1,1].set_xlabel('Service Type', labelpad=15)

plt.show()



# MACHINE LEARNING SETUP
y = df['Churn Value']
X = df.drop(['Churn Value', 'Churn Score', 'CLTV'], axis=1, errors='ignore')
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling (Crucial for Logistic Regression performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# MODEL TRAINING (With Balanced Class Weights)

# A. Logistic Regression (Added 'balanced' weight as requested)
lr_model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
lr_model.fit(X_train_scaled, y_train)

# B. Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)


# COMPARATIVE EVALUATION
def print_report(model, X_t, y_t, name):
 preds = model.predict(X_t)
 probs = model.predict_proba(X_t)[:, 1]
 print(f"\n--- {name} PERFORMANCE REPORT ---")
 print(f"ROC-AUC: {roc_auc_score(y_t, probs):.4f}")
 print(classification_report(y_t, preds))
 print_report(lr_model, X_test_scaled, y_test, "LOGISTIC REGRESSION")
 print_report(rf_model, X_test, y_test, "RANDOM FOREST")


# FINAL PERFORMANCE & FEATURE IMPORTANCE

# Precision-Recall Curve for Logistic Regression
plt.figure(figsize=(10, 6))
PrecisionRecallDisplay.from_estimator(lr_model, X_test_scaled, y_test, color="#3498db")
plt.title('Logistic Regression: Precision-Recall Curve\n(Balanced Class Weights)', fontsize=14, pad=30)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Feature Importance from Random Forest
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_10 = importances.sort_values(ascending=True).tail(10)

plt.figure(figsize=(12, 8))
top_10.plot(kind='barh', color='#8e44ad', edgecolor='black')
plt.title('Top 10 Drivers of Customer Churn\n(Random Forest Insights)', fontsize=15, pad=30)
plt.xlabel('Importance Score', labelpad=20)
plt.ylabel('Feature Name', labelpad=20)
plt.tight_layout()
plt.show()