#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df= pd.read_csv('nexcap.csv')
df


# In[10]:


df.head()


# In[12]:


df.tail()


# In[14]:


df.describe()


# In[16]:


df.info()


# In[18]:


X = df[['year']]
y = df['per_capita_income']


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)


# In[37]:


regressor =LinearRegression(fit_intercept=True)
regressor.fit(X_train,y_train)


# In[38]:


y_predict=regressor.predict(X_test)
print(y_predict)


# In[44]:


plt.scatter(y_test,y_predict)
plt.xlabel('actual')
plt.ylabel('predicted')


# In[58]:


predict_y_df=pd.DataFrame({'actual value':y_test,'predicted value':y_predict,'difference':y_test-y_predict})
predict_y_df[0:20]


# In[ ]:





# In[ ]:




