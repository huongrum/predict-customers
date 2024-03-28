#!/usr/bin/env python
# coding: utf-8

# In[9]:


pip install lazypredict


# In[11]:


# import library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
from sklearn import datasets 
from sklearn.utils import shuffle 
# Importing LazyRegressor 



# In[3]:


#import data
df_train = pd.read_csv('C:/MCI/test_MCI/data/churn-bigml-80.csv')
df_test = pd.read_csv('C:/MCI/test_MCI/data/churn-bigml-20.csv')


# In[127]:


# data type
df_train.info()


# # Data preprocessing

# In[4]:


#Delete unnecessary columns including 'State', 'International plan', 'Voice mail plan columns'
df_train_drop = df_train.drop(['State','International plan','Voice mail plan'], axis=1)
df_test_drop = df_test.drop(['State','International plan','Voice mail plan'], axis=1)


# In[5]:


#Separate the "Churn" column to make the label column
labels_train = df_train_drop.pop('Churn')
print(df_train_drop)
print(labels_train)
labels_test = df_test_drop.pop('Churn')
print(df_test_drop)
print(labels_test)


# In[6]:


# Normalize the Churn column to a number
le = LabelEncoder()
labels_train = le.fit_transform(labels_train)
labels_test = le.fit_transform(labels_test)
print(labels_train)
print(labels_test)


# In[7]:


# Since the data columns have a roughly bell-shaped histogram, we will use the StandardScaler() normalization method.
scaler = StandardScaler()
df_train_normal = pd.DataFrame(scaler.fit_transform(df_train_drop), columns=df_train_drop.columns)
df_test_normal = pd.DataFrame(scaler.fit_transform(df_test_drop), columns=df_test_drop.columns)
print(df_test_normal)
print(df_train_normal)


# In[8]:


#data train and data test
X_train = df_train_normal
y_train = labels_train
X_test = df_test_normal
y_test = labels_test


# In[16]:


#evaluated through models
from lazypredict.Supervised import LazyClassifier
# fitting data in LazyRegressor because 
# here we are solving Regression use case. 
clf = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None) 

# fitting data in LazyClassifier 
models, predictions = clf.fit(X_train, X_test, y_train, y_test) 
# lets check which model did better 
# on Breast Cancer Dataset 
print(models) 


# In[103]:


#Plot the columns in the train data set
import matplotlib.pyplot as plt
for column_name, column_data in df_train_drop.items():
    plt.hist(column_data, bins=5, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram chart for {}'.format(column_name))
    plt.show()


# In[104]:


#Plot the columns in the test data set
import matplotlib.pyplot as plt
for column_name, column_data in df_test_drop.items():
    plt.hist(column_data, bins=5, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram chart for {}'.format(column_name))
    plt.show()


# # Build and train model
# 
Here I choose two models: Logistic Regression and Random Forests to classify customers, then which model is more effective I will use that model to predict whether the customer is a potential customer or not.
# In[192]:





# ### Random Forests model

# In[194]:


# Creat a Random Forests model
model_Ra = RandomForestClassifier(n_estimators=100,
                                  max_depth=10,
                                  random_state=42)

# Train model
model_Ra.fit(X_train, y_train)

# Predict labels for test data
y_pred = model_Ra.predict(X_test)

# Evaluate the model on test data
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate recall
recall = recall_score(y_test, y_pred)

# Calculate F1-score
f1 = f1_score(y_test, y_pred)

# Print
print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1-score:", f1)
# transform y_pred to type data frame
y_pred = pd.DataFrame(y_pred)
print(y_pred)

From the above two models we can see that the Random Forests model is more effective than the Logistic Regression model, which is clearly shown through accuracy, recall coverage and f1-score value. Therefore, I will choose the Random Forests model to predict customers
# ### Predict customers using Random Forests model

# In[180]:


# Save the Random Forests model
model_filename = 'C:/MCI/test_MCI/model/mymodel.joblib'
joblib.dump(model_Ra, model_filename)

# Load model from file
loaded_model = joblib.load(model_filename)


# In[191]:


# Prediction
Account_length = int(input('Account length: '))
Area_code = int(input('Area code: '))
Number_vmail_messages = int(input('Number vmail messages: '))
Tota_day_minutes = float(input('Total day minutes: '))
Total_day_calls = int(input('Total day callse: '))
Total_day_charge = float(input('Total day charge: '))
Total_eve_minutes = float(input('Total eve minutes: '))
Total_eve_calls = int(input('Total eve calls: '))
Total_eve_charge = float(input('Total eve charge: '))
Total_night_minutes = float(input('Total night minutes: '))
Total_night_calls = int(input('Total night calls: '))
Total_night_charge = float(input('Total night charge: '))
Total_intl_minutes = float(input('Total intl minutes: '))
Total_intl_calls = int(input('Total intl calls: '))
Total_intl_charge= float(input('Total intl charge: '))
Customer_service_calls = int(input('Customer service calls: '))

customer_data = [Account_length,
                      Area_code,
                      Number_vmail_messages,
                      Tota_day_minutes,
                      Total_day_calls,
                      Total_day_charge,
                      Total_eve_minutes,
                      Total_eve_calls,
                      Total_eve_charge,
                      Total_night_minutes,
                      Total_night_calls,
                      Total_night_charge,
                      Total_intl_minutes,
                      Total_intl_calls,
                      Total_intl_charge,
                      Customer_service_calls]


predicted_labels = loaded_model.predict([customer_data])

# Print prediction results (churn = 0 --> False, churn = 1 --> True)
print('-------------------------')
if predicted_labels[0] == 1:
    print("True")
else:
    print("False")


# In[ ]:




