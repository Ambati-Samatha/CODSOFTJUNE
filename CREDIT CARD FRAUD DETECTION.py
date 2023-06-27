#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats
import numpy as np
import pandas as pd


# In[3]:


import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('creditcard.csv')


# In[4]:


data.info()


# In[5]:


data.head()


# In[6]:


print(data.shape)


# In[7]:


data.isnull().values.any()


# In[8]:


fraud = data[data['Class'] == 1] 
valid = data[data['Class'] == 0]


# In[9]:


print(fraud.shape,valid.shape)


# In[10]:


# dividing the X and the Y from the dataset 
#X = data.drop(['Class'], axis = 1)
X = data[['Time','Amount']]
Y = data["Class"] 
print(X.shape) 
print(Y.shape) 
# getting just the values for the sake of processing  
# (its a numpy array with no columns) 
xData = X.values 
yData = Y.values 


# In[11]:


# Using Skicit-learn to split data into training and testing sets 
from sklearn.model_selection import train_test_split 
# Split the data into training and testing sets 
xTrain, xTest, yTrain, yTest = train_test_split( 
        xData, yData, test_size = 0.2, random_state = 40)


# In[13]:


from sklearn.ensemble import RandomForestClassifier 
# random forest model creation 
rfc = RandomForestClassifier() 
rfc.fit(xTrain, yTrain) 
# predictions 
yPred = rfc.predict(xTest) 


# In[14]:


from sklearn.metrics import accuracy_score  
acc = accuracy_score(yTest, yPred) 
print("The accuracy of Render forest is {}".format(acc)) 


# In[15]:


from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB()
gnb.fit(xTrain, yTrain)
yPred=gnb.predict(xTest)
acc = accuracy_score(yTest, yPred) 
print("The accuracy of Naive Bayes is {}".format(acc))


# In[17]:


from sklearn.dummy import DummyClassifier
dummy = DummyClassifier()
dummy.fit(xTrain, yTrain)
yPred=dummy.predict(xTest)
acc = accuracy_score(yTest, yPred) 
print("The accuracy of Dummy Classifier is {}".format(acc)) 


# In[18]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(xTrain, yTrain)
yPred=svm.predict(xTest)
acc = accuracy_score(yTest, yPred) 
print("The accuracy of SVM is {}".format(acc)) 


# In[19]:


print(fraud.shape)


# In[20]:


# Lets shuffle the data before creating the subsamples

data1 = data.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_data1 = data1.loc[data1['Class'] == 1]
non_fraud_data1 = data1.loc[data1['Class'] == 0][:492]

normal_distributed_data1 = pd.concat([fraud_data1, non_fraud_data1])

# Shuffle dataframe rows
new_data1 = normal_distributed_data1.sample(frac=1, random_state=42)

new_data1.head()


# In[21]:


print('Distribution of the Classes in the subsample dataset')
print(new_data1['Class'].value_counts()/len(new_data1))


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot('Class', data=new_data1)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()


# In[22]:


# dividing the X and the Y from the dataset 
#X = data.drop(['Class'], axis = 1)
X = new_data1[['Time','Amount']]
Y = new_data1["Class"] 
print(X.shape) 
print(Y.shape) 
# getting just the values for the sake of processing  
# (its a numpy array with no columns) 
xData = X.values 
yData = Y.values 


# In[23]:


# Using Skicit-learn to split data into training and testing sets 
from sklearn.model_selection import train_test_split 
# Split the data into training and testing sets 
xTrain, xTest, yTrain, yTest = train_test_split( 
        xData, yData, test_size = 0.2, random_state = 40)


# In[24]:


from sklearn.ensemble import RandomForestClassifier 
# random forest model creation 
rfc = RandomForestClassifier() 
rfc.fit(xTrain, yTrain) 
# predictions 
yPred = rfc.predict(xTest) 


# In[25]:


from sklearn.metrics import accuracy_score  
acc = accuracy_score(yTest, yPred) 
print("The accuracy of Render forest is {}".format(acc)) 

