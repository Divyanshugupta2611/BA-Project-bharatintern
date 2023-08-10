#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:


df = pd.read_csv(r'C:\Users\Divyanshu\OneDrive\Documents\Desktop\BA\Empdata.csv')
df.head()


# In[24]:


df = pd.read_csv('C:\\Users\\Divyanshu\\OneDrive\\Documents\\Desktop\\BA\\Empdata.csv')
df.head()


# In[25]:


df.shape


# In[26]:


df.describe()


# In[27]:


df.isnull().sum()


# In[28]:


attrition_count = pd.DataFrame(df['Attrition'].value_counts())
attrition_count


# In[29]:


plt.pie(attrition_count['Attrition'] , labels = ['No' , 'Yes'] , explode = (0.2,0))


# In[30]:


sns.countplot(df['Attrition'])


# In[31]:


df.drop(['EmployeeCount' , 'EmployeeNumber'] , axis = 1)


# In[32]:


attrition_dummies = pd.get_dummies(df['Attrition'])
attrition_dummies.head()


# In[33]:


df = pd.concat([df, attrition_dummies] , axis = 1)
df.head()


# In[34]:


df = df.drop(['Attrition' , 'No'] , axis = 1)
df.head()


# In[35]:


sns.barplot(x = 'Gender' , y = 'Yes', data = df)


# In[36]:


sns.barplot(x = 'Department', y = 'Yes', data = df)


# In[37]:


sns.barplot(x = 'BusinessTravel', y = 'Yes', data = df)


# In[39]:


plt.figure(figsize = (10,6))
sns.heatmap(df.corr())


# # data preprocessing

# In[ ]:


from sklearn.preprocessing import LabelEncoder
for column in df.columns:
    if df[column].dtype==np.number:
        continue
    else:
        df[column]=LabelEncoder().fit_transform(df[column])


# # Model Building

# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
x  = df.drop(['Yes'], axis = 1)
y = df['Yes']
x_train, x_test , y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)

x_train.head()


# In[41]:


rf.fit(x_train, y_train)


# In[42]:


rf.score(x_train, y_train)


# In[43]:


rf.score(x_train, y_train)


# # Predicting for x_test

# In[44]:


pred = rf.predict(x_test)


# In[45]:


from sklearn.metrics import accuracy_score


# In[46]:


accuracy_score(y_test, pred)


# In[ ]:




