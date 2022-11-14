#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


# In[2]:


# get working directory
from pathlib import Path
print(Path.cwd())


# In[3]:


# or another way...
import os
cwd = os.getcwd()
print(cwd)


# In[4]:


# change working dir
os.chdir("C:\\Users\\dpapa") 

cwd = os.getcwd()
print(cwd)


# In[5]:


# load dataset 
df1 = pd.read_csv(r"C:\Users\dpapa\winequality-red.csv")
df2 = pd.read_csv(r"C:\Users\dpapa\winequality-white.csv")


# In[6]:


df1


# In[7]:


df2


# In[8]:


# append df & df2 into one single dataset
df = pd.concat([df1,df2])


# In[9]:


df


# In[10]:


df.head(10)


# In[11]:


df.shape


# In[12]:


# look for missing values
df.info() 


# In[13]:


# find which column has missing values
print(df.isnull().sum().sort_values(ascending=False).to_string())


# In[14]:


# number of unique values in each feature. 
df.nunique()


# In[15]:


# histogram with specified number of bins
df.hist(bins=10, figsize=(15, 15))


# In[16]:


# distribution for "quality"
op_count = df['quality'].value_counts()
plt.figure(figsize=(10,5))
sns.barplot(x=op_count.index,y= op_count.values)
plt.title('Class')
plt.ylabel('occurances by class', fontsize=12)
plt.xlabel('Class', fontsize=12)
plt.show()


# In[17]:


# some general statistics for data features/columns after removing outliers
df.select_dtypes(exclude=['object']).describe().round(decimals=2).transpose()


# In[18]:


# correlation matrix and heatmap for feature selection
corr=df.corr()
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)


# In[19]:


print("Most positively correlated features with the target")
corr.sort_values(['quality'], ascending=False, inplace=True)
corr.quality


# In[20]:


columns =list(df.columns)
columns


# ### Removed all features that have negative correlation with target variable "quality"  

# In[21]:


#select features X = Bland Chromatin and target Y= class
X = df[[
'alcohol','citric acid', 
'free sulfur dioxide',     
'sulphates',          
'pH' ]].values
Y = df[['quality']].values


# In[22]:


# split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=7)


# # Random Forest Classifier

# In[23]:


model = RandomForestClassifier (max_depth=100, random_state=0, n_estimators=10)


# In[24]:


# train model with data
model.fit(X_train,y_train.ravel())


# In[25]:


# make predictions
y_pred = model.predict(X_test)
df2 = pd.DataFrame(y_pred, columns = ['y-pred'])
vertical_concat = pd.concat([df2, pd.DataFrame(y_test,columns=['label'])], axis = 1 )
vertical_concat.head(10)


# In[26]:


# comfusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)


# In[27]:


# accuracy 
score = model.score(X_test, y_test)
print(score*100,'%')


# In[28]:


import pickle
# Open a file and use dump() 
with open('wine_quality.pkl', 'wb') as file: 
      
    # A new file will be created 
    pickle.dump(model, file)


# In[ ]:




