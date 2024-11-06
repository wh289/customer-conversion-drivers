#!/usr/bin/env python
# coding: utf-8

# # Telco Churn Prediction

# ### Objective: 
# Identify factors that contribute to customer churn and build a predictive model to anticipate which customers are likely to leave the teleco.
# 
# ### Target Variable: 
# In the dataset, the churn column is handily labeled for us. This column will be used as the y variable that we predict with the model.

# ### Index:
# - Loading libraries and data
# - Data Exploration
# - Data Cleaning
# - Feature engineering
# - Data Visualization
# - Data Preprocessing
# 
# #### Machine Learning Model Evaluations and Predictions
# - Random Forest
# - Logistic Regression
# - Gradient Boosting Classifier
# 

# ## Questions To Answer
# - Is there a difference in churn rate based on gender?
# - Are certain products better/worse for churn rates?
# - How much impact does billing type have?
# - Is there a tenure after which churn becomes much less likely?
# 
# Any other questions that arise as we go through the data

# In[134]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[135]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report


# In[136]:


df = pd.read_csv("/workspace/customer-conversion-drivers/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# ## Data Exploration
# 

# In[137]:


pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows

# Display the DataFrame
print(df.head())

# Optionally reset options if you want to revert back to default settings
pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')


# We can see that certain columns don't seem to be in the correct format, for example TotalCharges and MonthlyCharges should both be numeric

# In[141]:


df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
## Creating a list of numeric columns
numeric_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']
numeric_cols


# In[142]:


df.info()


# We can see that after converting to numeric and Nulling rows that couldn't be converted, we now have 11 NULL rows in TotalCharges

# These missing rows are now filled with the mean value

# In[143]:


df.fillna(df["TotalCharges"].mean())
df.drop(columns=['customerID'], inplace = True)


# Checking for dupes

# In[144]:


duplicates = df.duplicated().sum()
print(f'Duplicate rows: {duplicates}')


# Checking for outliers

# In[145]:


# Calculate IQR
Q1 = df['MonthlyCharges'].quantile(0.25)
Q3 = df['MonthlyCharges'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['MonthlyCharges'] < lower_bound) | (df['MonthlyCharges'] > upper_bound)]
print(outliers)


# In[146]:


print(f"Lower bound: {lower_bound}")
print(f"Upper bound: {upper_bound}")


# In[147]:


df['MonthlyCharges'].describe()


# ## Data cleaning

# Cleaning Up MultipleLines column

# In[148]:


df['MultipleLines'] =  df['MultipleLines'].apply(lambda x: True if x == 'Yes' else False)


# Cleaning columns so that the relevant columns are Bools rather than objects

# In[149]:


binary_int_columns = ['SeniorCitizen']

# Convert 0/1 columns to bool
df[binary_int_columns] = df[binary_int_columns].astype(bool)


# In[150]:


# Convert "yes"/"no" to bool
obj_columns = ['Churn','Partner','Dependents','PhoneService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']

def map_to_boolean(value):
    if value == 'Yes':
        return True
    elif value == 'No':
        return False
    else:  # To Handle 'No Service' and any other unexpected values
        return False

for col in obj_columns:
    df[col] = df[col].apply(map_to_boolean)


# One Hot encoding

# In[151]:


df.info()


# In[152]:


# One hot encoding
df = pd.get_dummies(df, columns=['gender','InternetService','Contract','PaperlessBilling','PaymentMethod'])


# We are left with either bool or int columns and no nulls

# In[153]:


df.info()


# # Feature Engineering

# In[ ]:





# # Data Visualisation

# In[154]:


# 1. Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['MonthlyCharges'], bins=10, kde=True)
plt.title('Distribution of Monthly Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Frequency')
plt.show()

# 2. Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['TotalCharges'])
plt.title('Boxplot of Total Charges')
plt.xlabel('Total Charges')
plt.show()

# 3. Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='MonthlyCharges', y='TotalCharges', data=df)
plt.title('Scatter Plot of Monthly Charges vs. Total Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Total Charges')
plt.show()


# In[164]:


correlation_matrix = df.corr()['Churn'].sort_values(ascending = False)
correlation_matrix = correlation_matrix.drop('Churn')


# In[170]:


plt.figure(figsize=(20, 6))
sns.barplot(x=correlation_matrix.index, y=correlation_matrix.values, palette='coolwarm')
plt.title('Correlation of Features with Churn (Excluding Churn)')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)  
plt.show()


# # Data Preprocessing

# In[171]:


X = df.drop('Churn', axis = 1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Machine Learning Modelling and Predictions

# Model training

# In[172]:


model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)


# Model predictions

# In[173]:


model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[179]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
conf_matrix


# Looking at Feature Importance

# In[180]:


# Feature importants
importance = model.get_booster().get_score(importance_type='weight')
importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance from XGBoost')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


# # Hyperparam tuning

# In[181]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize the model
xgb_model = XGBClassifier(eval_metric='logloss')

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           scoring='accuracy', cv=5, n_jobs=-1, verbose=1)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)


# Cross Validation

# In[182]:


from sklearn.model_selection import cross_val_score

# Initialize the model
model_cv = XGBClassifier(eval_metric='logloss')

# Perform cross-validation
cv_scores = cross_val_score(model_cv, X, y, cv=5, scoring='accuracy')

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())


# In[183]:


best_params


# In[184]:


print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())


# In[185]:


# Best parameters from GridSearchCV
best_params = {
    'colsample_bytree': 1.0,
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 100,
    'subsample': 0.8
}

# Initialize the model with the best parameters
model = XGBClassifier(**best_params, eval_metric='logloss')

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# In[ ]:




