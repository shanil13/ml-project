#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[4]:


dataset = pd.read_csv(r'C:\Users\shani\Downloads\gld_price_data.csv')


# In[5]:


dataset.head()


# In[6]:


print(dataset.shape)


# In[7]:


dataset.info()
# all the columns are float type only date is categorical type


# In[8]:


dataset.describe()


# In[9]:


print(dataset.isnull().sum())
#as we can see there are no null values in our dataset


# In[10]:


duplicate_rows = dataset[dataset.duplicated()]
print("Number of duplicate rows: ", duplicate_rows.shape)
#as we can see there are no duplicate rows


# In[11]:


plt.subplots(figsize=(8,8))
sns.heatmap(dataset.corr(), cbar=True, square=True, fmt='.2f', annot=True, annot_kws = {'size':10}, cmap='Blues')


# In[12]:


sns.jointplot(x =dataset['SLV'], y = dataset['GLD'], color = 'deeppink')
# Now we check the relation of SLV with GLD variable
# as we can see that the conclusion we made using the heatmap is true.


# In[13]:


sns.jointplot(x =dataset['USO'], y = dataset['GLD'], color = 'purple')
#Now we check the relation of USO with GLD variable


# In[14]:


sns.jointplot(x =dataset['SPX'], y = dataset['GLD'], color = 'crimson')
#Now we check the relation of SPX with GLD variable


# In[15]:


sns.jointplot(x =dataset['EUR/USD'], y = dataset['GLD'], color = 'darkorange')
#Now we check the relation of EUR?USD with GLD variable


# In[16]:


sns.distplot(dataset['GLD'], color = 'blue')
# Lets Check our target variable


# In[17]:


x = dataset.drop(['Date', 'GLD'], axis=1)
y = dataset['GLD']


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape)


# In[19]:


def fit_predict(model1):
    model = model1
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print('RSS: ', metrics.r2_score(y_test, y_predict))
    print('MAE :', metrics.mean_absolute_error(y_test, y_predict))
    plt.scatter(y_test, y_predict)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Price of Gold')
    plt.show()
    plt.scatter(y_test, y_predict, c='crimson')
    p1 = max(max(y_predict), max(y_train))
    p2 = min(min(y_predict), min(y_train))
    plt.plot([p1,p2], [p1,p2], 'b-')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Price')
    plt.show()


# In[20]:


from sklearn.linear_model import LinearRegression
fit_predict(LinearRegression())


# In[21]:


from sklearn.ensemble import RandomForestRegressor
fit_predict(RandomForestRegressor(n_estimators = 100, random_state = 0))


# In[22]:


from xgboost import XGBRegressor
fit_predict(XGBRegressor())


# In[23]:


model = XGBRegressor()                                    # using XGB to make predictions for our test case
model.fit(x_train, y_train)

test_case_list = [1254.25, 70.2527, 18.280001, 0.9232]    #creating a list of random price of independent variables
test_case_df = pd.DataFrame(test_case_list)               # converting the list into a dataframe
test_case = test_case_df.T                                # the shape of current df is (4,1) and the required shape is (1,4)
test_case_prediction = model.predict(test_case)           # predicting the price of GOLD for test case
print('Price of Gold would be:', test_case_prediction)    #printing the result


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




