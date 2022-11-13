#!/usr/bin/env python
# coding: utf-8

# In[167]:


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


# In[168]:


# get working directory
from pathlib import Path
print(Path.cwd())


# In[169]:


# or another way...
import os
cwd = os.getcwd()
print(cwd)


# In[170]:


# change working dir
os.chdir("C:\\Users\\dpapa") 

cwd = os.getcwd()
print(cwd)


# In[171]:


# load dataset 
df1 = pd.read_csv(r"C:\Users\dpapa\winequality-red.csv")
df2 = pd.read_csv(r"C:\Users\dpapa\winequality-white.csv")


# In[172]:


df1


# In[173]:


df2


# In[174]:


# append df & df2 into one single dataset
df = pd.concat([df1,df2])


# In[175]:


df


# In[176]:


df.head(10)


# In[177]:


df.shape


# In[178]:


# look for missing values
df.info() 


# In[179]:


# find which column has missing values
print(df.isnull().sum().sort_values(ascending=False).to_string())


# In[180]:


# number of unique values in each feature. 
df.nunique()


# In[181]:


# histogram with specified number of bins
df.hist(bins=10, figsize=(15, 15))


# In[184]:


# distribution for "quality"
op_count = df['quality'].value_counts()
plt.figure(figsize=(10,5))
sns.barplot(x=op_count.index,y= op_count.values)
plt.title('Class')
plt.ylabel('occurances by class', fontsize=12)
plt.xlabel('Class', fontsize=12)
plt.show()


# In[185]:


# some general statistics for data features/columns after removing outliers
df.select_dtypes(exclude=['object']).describe().round(decimals=2).transpose()


# In[186]:


# correlation matrix and heatmap for feature selection
corr=df.corr()
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)


# In[191]:


print("Most positively correlated features with the target")
corr.sort_values(['quality'], ascending=False, inplace=True)
corr.quality


# In[192]:


columns =list(df.columns)
columns


# ### Removed all features that have negative correlation with target variable "quality"  

# In[202]:


#select features X = Bland Chromatin and target Y= class
X = df[['quality',
'alcohol','citric acid', 
'free sulfur dioxide',     
'sulphates',          
'pH' ]].values
Y = df[['quality']].values


# In[203]:


# split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=7)


# # Random Forest Classifier

# In[204]:


model = RandomForestClassifier (max_depth=100, random_state=0, n_estimators=10)


# In[205]:


# train model with data
model.fit(X_train,y_train.ravel())


# In[206]:


# make predictions
y_pred = model.predict(X_test)
df2 = pd.DataFrame(y_pred, columns = ['y-pred'])
vertical_concat = pd.concat([df2, pd.DataFrame(y_test,columns=['label'])], axis = 1 )
vertical_concat.head(10)


# In[207]:


# comfusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)


# In[208]:


# accuracy 
score = model.score(X_test, y_test)
print(score*100,'%')


# In[210]:


import pickle
# Open a file and use dump() 
with open('wine_quality.pkl', 'wb') as file: 
      
    # A new file will be created 
    pickle.dump(model, file)


# In[ ]:




