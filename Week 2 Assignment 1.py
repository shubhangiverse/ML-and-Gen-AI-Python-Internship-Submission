# SHUBHANGI SINHA
# 19101172025

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Q1 Load the dataset and display first five records

df = pd.read_csv("Netflix_User_Analytics.csv")

print("First 5 Records")
print(df.head())

# Q2 Determine number of rows and columns

print("\nDataset Shape")
print(df.shape)

# Q3 Display all column names

print("\nColumn Names")
print(df.columns)

# Q4 Identify numerical and categorical features

print("\nNumerical Features")
print(df.select_dtypes(include=np.number).columns)

print("\nCategorical Features")
print(df.select_dtypes(include='object').columns)

# Q5 Check for missing values

print("\nMissing Values")
print(df.isnull().sum())

# Q6 Calculate average age of users

print("\nAverage Age")
print(df["Age"].mean())

# Q7 Determine average watch hours per week

print("\nAverage Watch Hours Per Week")
print(df["WatchHoursPerWeek"].mean())

# Q8 Find average monthly spending

print("\nAverage Monthly Spending")
print(df["MonthlySpend"].mean())

# Q9 Count users in each subscription category

print("\nSubscription Category Counts")
print(df["SubscriptionType"].value_counts())

# Q10 Determine percentage of users who renewed subscriptions

renewal_percentage = (
    (df["SubscriptionRenewed"] == "Yes").sum()
    / len(df)
) * 100

print("\nSubscription Renewal Percentage")
print(renewal_percentage)

# Q11 Convert categorical features into numerical form

df_encoded = df.copy()

encoder = LabelEncoder()

for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = encoder.fit_transform(df_encoded[col])

print("\nEncoded Dataset")
print(df_encoded.head())

# Q12 Define feature set X and target variable y

X = df_encoded.drop("SubscriptionRenewed", axis=1)
y = df_encoded["SubscriptionRenewed"]

# Q13 Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Q14 Train Decision Tree model

decision_tree = DecisionTreeClassifier(random_state=42)

decision_tree.fit(X_train, y_train)

# Q15 Evaluate Decision Tree using accuracy

dt_predictions = decision_tree.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\nDecision Tree Accuracy")
print(dt_accuracy)

# Q16 Generate confusion matrix

cm = confusion_matrix(y_test, dt_predictions)

print("\nConfusion Matrix")
print(cm)

# Q17 Train KNN classifier with K = 5

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

knn_predictions = knn.predict(X_test)

# Q18 Compare accuracy of KNN and Decision Tree

knn_accuracy = accuracy_score(y_test, knn_predictions)

print("\nKNN Accuracy")
print(knn_accuracy)

print("\nAccuracy Comparison")
print("Decision Tree:", dt_accuracy)
print("KNN:", knn_accuracy)

# Q19 Train Linear Regression model to predict monthly spending

X_reg = df_encoded.drop("MonthlySpend", axis=1)
y_reg = df_encoded["MonthlySpend"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

linear_model = LinearRegression()

linear_model.fit(X_train_reg, y_train_reg)

# Q20 Predict monthly spending for a new user

new_user = [X_reg.iloc[0].values]

predicted_spending = linear_model.predict(new_user)

print("\nPredicted Monthly Spending")
print(predicted_spending[0])

# Business Reflection Question 1
# Factors that may influence subscription renewal are
# Age, WatchHoursPerWeek, MonthlySpend,
# SubscriptionType, DevicesUsed and FavoriteGenre

# Business Reflection Question 2
# Subscription renewal is a classification problem
# because the output is categorical
# Yes or No

# Business Reflection Question 3
# Monthly spending is a regression problem
# because the output is a continuous numerical value

# Business Reflection Question 4
# The algorithm with higher accuracy
# between Decision Tree and KNN
# performed better for renewal prediction

# Business Reflection Question 5
# Netflix can use predictions to identify users
# likely to leave the platform, offer discounts,
# provide personalized recommendations and
# improve customer retention strategies