#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[2]:


dataset = pd.read_csv(r'C:\Users\shani\Downloads\Final Project 1.csv')


# In[3]:


dataset.head()


# In[4]:


print(dataset.shape)


# In[5]:


print(dataset.isnull().sum())


# In[6]:


dataset.info()


# In[7]:


duplicate_rows_df = dataset[dataset.duplicated()]
print("Number of duplicate rows: ", duplicate_rows_df.shape )


# In[8]:


dataset = dataset.drop_duplicates()
print(dataset.shape)


# In[9]:


dataset.describe()


# In[10]:


fig = plt.figure(figsize = (5,3))
sns.barplot(x = 'quality', y = 'fixed acidity', data = dataset)
#Here we see that fixed acidity does not give any specification to classify the quality.


# In[11]:


fig = plt.figure(figsize = (5,3))
sns.barplot(x = 'quality', y = 'volatile acidity', data = dataset)
#Here we see that higher quality wine has lower volatile acidity


# In[12]:


fig = plt.figure(figsize = (5,3))
sns.barplot(x = 'quality', y = 'citric acid', data = dataset)
#Composition of citric acid go higher as we go higher in the quality of the wine


# In[13]:


fig = plt.figure(figsize = (5,3))
sns.barplot(x = 'quality', y = 'residual sugar', data = dataset)
# Residual Sugar does not have a significant effect on the quality of wine


# In[14]:


fig = plt.figure(figsize = (5,3))
sns.barplot(x = 'quality', y = 'chlorides', data = dataset)
#Composition of chloride also go down as we go higher in the quality of the wine


# In[15]:


fig = plt.figure(figsize = (5,3))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = dataset)


# In[16]:


fig = plt.figure(figsize = (5,3))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = dataset)


# In[17]:


fig = plt.figure(figsize = (5,3))
sns.barplot(x = 'quality', y = 'sulphates', data = dataset)


# In[18]:



fig = plt.figure(figsize = (5,3))
sns.barplot(x = 'quality', y = 'alcohol', data = dataset)
#Alcohol level also goes higher as te quality of wine increases


# In[21]:


#heatmap
plt.subplots(figsize=(10,10))
sns.heatmap(dataset.corr(), cbar=True, square=True, fmt='.1f', annot=True, annot_kws = {'size':8}, cmap='Blues')


# In[ ]:


print(dataset['quality'].value_counts())


# In[ ]:


sns.countplot(x=dataset['quality'])


# In[ ]:


def cat(x):
    if x < 6.5:
        return 'poor'
    else:
        return 'good'
dataset['quality']=dataset['quality'].apply(lambda x: cat(x))


# In[ ]:


print(dataset['quality'].value_counts())


# In[ ]:


sns.countplot(dataset['quality'])


# In[ ]:


label_quality = LabelEncoder()

dataset['quality'] = label_quality.fit_transform(dataset['quality'])


# In[ ]:


print(dataset['quality'].value_counts())


# In[ ]:


x = dataset.drop(['quality'], axis=1)
y = dataset['quality']


# In[ ]:


#feature scaling
sc = StandardScaler()
x = sc.fit_transform(x)


# In[ ]:



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
print(x_train.shape, x_test.shape)


# In[ ]:


def fit_predict(model1):
    model = model1
    model.fit(x_train, y_train)
    y_pred=model.predict(x_test)
    print("Accuracy score: ",(accuracy_score(y_test, y_pred)*100))
    print("\nConfusion Matirx")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report")
    print(classification_report(y_test,y_pred))


# In[ ]:


#logistic regression
from sklearn.linear_model import LogisticRegression
fit_predict(LogisticRegression())


# In[ ]:


#knn
from sklearn.neighbors import KNeighborsClassifier
fit_predict(KNeighborsClassifier(n_neighbors=7))


# In[ ]:


#decision tree
from sklearn.tree import DecisionTreeClassifier
fit_predict(DecisionTreeClassifier(max_depth=2))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
fit_predict(RandomForestClassifier(n_estimators=200, random_state=0))


# In[ ]:


from sklearn.svm import SVC
fit_predict(SVC())


# In[ ]:


model=SVC()# we are going to use SVC to make predictions for our test case
model.fit(x_train, y_train)

test_case_list = [7.4, 0.36, 0.3, 1.8, 0.074, 17, 24, 0.99419, 3.24, 0.7, 13.4]
#creating a list of random values for chemical compositions of wine

test_case_df = pd.DataFrame(test_case_list)               # converting the list into a dataframe
test_case = test_case_df.T                                # the shape of current df is (11,1) and the required shape is (1,11)
test_case_scaled = sc.transform(test_case)                # applying standard scaler to the test case
test_case_prediction = model.predict(test_case_scaled)    # predicting the quality for test case
if(test_case_prediction == 0):                            # generating the result.
    print('The Quality of wine is good.')
else:
    print('The Quality of wine is not up to the mark.')


# In[ ]:





# In[ ]:





# In[ ]:




