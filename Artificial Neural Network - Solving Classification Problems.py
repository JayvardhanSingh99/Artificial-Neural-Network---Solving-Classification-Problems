#!/usr/bin/env python
# coding: utf-8

# # Artificial Neural Network - Solving Classification Problems

# Artificial Neural Networks(ANN) are part of supervised machine learning where we will be having input as well as corresponding output present in our dataset. Our whole aim is to figure out a way of mapping this input to the respective output. ANN can be used for solving both regression and classification problems.

# **1. Importing the required libraries**

# In[20]:


import pandas as pd
import numpy as np
import tensorflow as tf


# **2. Importing the Data set**

# In[21]:


bank_df = pd.read_csv('bank.csv', delimiter=';')


# In[22]:


bank_df.head()


# In[23]:


# We have a dataset where we are having 17 dimensions in total and 4521 record
bank_df.shape


# * Here our main goal is to create an artificial neural network that will take into consideration all independent variables(first 16) and based on that will predict if our customer is going to exit the bank or not(Exited is dependent variable here).

# **4. Generating Matrix of Features for X (Independent variables)**

# In[24]:


X = bank_df.iloc[:, 0:-1].values


# In[25]:


X


# **5. Generating Dependent Variable Vector (Y)**

# In[26]:


Y = bank_df.iloc[:, -1].values


# In[27]:


Y


# * Feature engineering is a phase where we either generate new variables from existing ones or modify existing variables so as to use them in our machine learning model. We need to convert those string values into their numerical equivalent without losing their significance using Encoding.

# **6. Use Label Encoding for categorical variable(Y) to yes and no or 1 and 0**

# In[28]:


#encoding output variable y: subscribed or not
from sklearn.preprocessing import LabelEncoder


# In[29]:


LE1 = LabelEncoder()
Y = np.array(LE1.fit_transform(Y))


# In[30]:


Y


# **7. Use Label Encoding for features/categorical variable(X)**

# In[31]:


#using label encoder for marital status, default, housing, loan, contact and poutcomme
X[:,2] = np.array(LE1.fit_transform(X[:,2]))
X[:,4] = np.array(LE1.fit_transform(X[:,4]))
X[:,6] = np.array(LE1.fit_transform(X[:,6]))
X[:,7] = np.array(LE1.fit_transform(X[:,7]))
X[:,8] = np.array(LE1.fit_transform(X[:,8]))
X[:,15] = np.array(LE1.fit_transform(X[:,15]))


# In[32]:


X[0]


# **8. Using One Hot Encoding to convert job, education and month so that machine learning does not assume higher number is important. Using column Transformer to pick a particular column**

# In[33]:


from sklearn.compose import ColumnTransformer


# In[43]:


from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[1])], remainder="passthrough")
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[14])], remainder="passthrough")
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[24])], remainder="passthrough")
X = np.array(ct.fit_transform(X))


# In[44]:


X[0]


# **9. Splitting dataset into train test**

# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# **10. Perform Feature Scaling**

# * Feature scaling: sometimes certain variables can have high values and some low, without scaling the high value variable can dominate and low value variable may be neglected, so feature scaling is done.
# 
# * Feature scaling before or after split: should be done after splitting into train and test sets, otherwise it might result in information leakage on test set and it neglects the purpose of testing dataset.
# 
# * Feature scaling can be done using Standardization (everything converted between -3 to +3) and Normalization (-1 to +1)
# 
# * Use Normalization is used when data is normally distributed
# * Use Standardization is a universal technique and can be used even if data is not normally distributed

# In[46]:


from sklearn.preprocessing import StandardScaler


# In[47]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# **11. Initializing Artificial Neural Network**

# In[49]:


ann = tf.keras.models.Sequential()


# **12. Creating Hidden Layers**

# In[40]:


#Create First Hidden Layers
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))


# In[99]:


#Create Second Hidden Layers
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))


# **13. Creating Output Layer**

# In[100]:


ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


# **14. Compiling Artificial Neural Network**

# In[103]:


ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])


# **15.Fitting Artificial Neural Network**

# In[104]:


ann.fit(X_train,Y_train,batch_size=32,epochs=100)


# **16. Save your created neural network**

# In[ ]:


ann.save("ANN.h5")

