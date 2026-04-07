# 📉 Telco Churn: 
  # Why Customers Leave & How to Reduce It
### **A Data Science Project on Predictive Customer Retention**

Hi 👋 I'm **Eldho Joshy**, a Data Science student. This project focuses on understanding the underlying patterns of customer behavior and building a machine learning pipeline to predict churn before it happens.

---

## 💡 The Business Problem
In the telecom industry, the cost of acquiring a new customer is **5x higher** than retaining an existing one. High churn rates directly impact revenue and long-term sustainability. 

The goal of this project is to build a **"Risk Radar"** that identifies at-risk customers, allowing businesses to intervene with targeted retention strategies.

---

## 🔍 Key Insights (Exploratory Data Analysis)
Before building the models, I performed deep EDA to understand the "signals" of churn:

* **The 6-Month Window:** Most churn occurs within the first 6 months of a contract.
* **Contract Type:** Month-to-month users churn at a significantly higher rate than long-term subscribers.
* **Pricing Sensitivity:** Higher monthly charges are a primary driver for customers looking for alternatives.
* **Service Patterns:** Surprisingly, Fiber Optic users show higher churn than DSL users, suggesting potential service-specific friction.
* **Loyalty Factor:** Customers who stay past the 2-year mark become exponentially more stable.

<img width="1397" height="796" alt="dashboard-preview" src="https://github.com/eldho-joshy/customer-churn-prediction/blob/be5747a522ef5f1f476dbf7a36728f9190d97fbe/Visual%20output.png" />


---

## 🛠️ Technical Approach
To ensure the model was robust and unbiased, I implemented the following pipeline:

1.  **Data Cleaning:** Handled missing values in `Total Charges` and removed non-predictive identifiers.
2.  **Feature Engineering:** Applied **One-Hot Encoding** for categorical variables and **StandardScaler** for numerical normalization.
3.  **Handling Imbalance:** Used `class_weight='balanced'` and **Stratified Sampling** to ensure the model doesn't ignore the minority "Churn" class.
4.  **Model Selection:** Evaluated **Logistic Regression** (for interpretability) and **Random Forest** (for capturing non-linear patterns).

---

## 📊 Performance Comparison

| Metric | Logistic Regression | Random Forest |
| :--- | :---: | :---: |
| **ROC-AUC** | **~0.84** | ~0.83 |
| **Recall (Churn)** | **0.80** | 0.76 |
| **Accuracy** | ~72% | **~78%** |

> **Strategic Note:** I prioritized **Recall** over Accuracy. In churn prediction, it is more expensive to "miss" a customer who is about to leave (False Negative) than to accidentally offer a discount to a loyal one (False Positive).



---

## 🚀 Business Application: Risk Segmentation
Instead of a binary "Yes/No" output, I developed a **Risk Level Framework** to help marketing teams prioritize their budget:

* 🔴 **High Risk (>70%)** → Immediate retention offers (discounts, personalized outreach).
* 🟡 **Medium Risk (30–70%)** → Engagement & feedback surveys to improve satisfaction.
* 🟢 **Low Risk (<30%)** → Standard service and upselling opportunities.

---

## 👨‍💻 About Me
**Eldho Joshy**
* **LinkedIn:https://www.linkedin.com/in/eldho-joshy** 

---
*If you find this project helpful, feel free to ⭐ the repository!*
