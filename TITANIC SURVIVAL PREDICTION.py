#!/usr/bin/env python
# coding: utf-8

# In[19]:


#Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


dataset = pd.read_csv("train.csv")


# In[3]:


dataset.head()


# In[4]:


#Checking for missing values
dataset.isnull().sum()


# In[6]:


#Imputation of missing values
print('Median of Age column: %.2f' %(dataset["Age"].median(skipna=True)))
print('Percent of missing records in the Cabin column: %.2f%%' %((dataset['Cabin'].isnull().sum()/dataset.shape[0])*100))
print('Most common boarding port of embarkation: %s' %dataset['Embarked'].value_counts().idxmax())


# In[7]:


dataset["Age"].fillna(dataset["Age"].median(skipna=True), inplace=True)
dataset["Embarked"].fillna(dataset['Embarked'].value_counts().idxmax(), inplace=True)
dataset.drop('Cabin', axis=1, inplace=True)


# In[8]:


#Checking missing values after imputation of missing values
dataset.isnull().sum()


# In[9]:


#Dropping unnecessary columns
dataset.drop('PassengerId', axis=1, inplace=True)
dataset.drop('Name', axis=1, inplace=True)
dataset.drop('Ticket',  axis=1, inplace=True)


# In[10]:


#Creating categorical variable for traveling alone
dataset['TravelAlone']=np.where((dataset["SibSp"]+dataset["Parch"])>0, 0, 1)
dataset.drop('SibSp', axis=1, inplace=True)
dataset.drop('Parch', axis=1, inplace=True)


# In[11]:


dataset.head()


# In[12]:


#Number of rows and columns of train set
dataset.shape


# In[13]:


dataset.info()


# In[14]:


dataset.describe()


# In[20]:


#Count of passengers based on gender
sns.countplot('Sex',data=dataset)
dataset['Sex'].value_counts()


# In[16]:


#Effect of Sex feature on the survival rate
sns.barplot(x='Sex',y='Survived',data=dataset)
dataset.groupby('Sex',as_index=False).Survived.mean()


# In[17]:


sns.countplot(x='Survived', hue='Sex', data=dataset)


# In[21]:


#Count of passengers based on Pclass
sns.countplot('Pclass',data=dataset)
dataset['Pclass'].value_counts()


# In[22]:


#Effect of Pclass feature on the survival rate

sns.barplot(x='Pclass',y='Survived',data=dataset)
dataset.groupby('Pclass',as_index=False).Survived.mean()


# In[23]:


sns.countplot(x='Survived', hue='Pclass', data=dataset)


# In[24]:


#Count of the passengers based on Emabarked
sns.countplot('Embarked',data=dataset)
dataset['Embarked'].value_counts()


# In[25]:


#Count of passengers based on TravelAlone
sns.countplot('TravelAlone',data=dataset)
dataset['TravelAlone'].value_counts()


# In[26]:


#Analysis of Survived feature
sns.countplot('Survived',data=dataset)
dataset['Survived'].value_counts()


# In[27]:


#Correleation matrix
dataset.corr()


# In[28]:


#Import label encoder
from sklearn import preprocessing
  
#label_encoder object knows how to understand word labels
label_encoder = preprocessing.LabelEncoder()
  
#Encode labels in column Sex and Embarked
dataset['Sex']= label_encoder.fit_transform(dataset['Sex'])
dataset['Embarked']= label_encoder.fit_transform(dataset['Embarked'])


# In[29]:


dataset.head()


# In[30]:


#Setting the value for dependent and independent variables
X = dataset.drop('Survived', 1)
y = dataset.Survived


# In[31]:


#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)


# In[32]:


#Fitting the Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)


# In[33]:


y_pred = lr_model.predict(X_test)
y_pred


# In[34]:


#Accuracy of the Logistic Regression model
from sklearn.metrics import accuracy_score
print('Accuracy of the model: {:.2f}'.format(accuracy_score(y_test, y_pred)*100))

