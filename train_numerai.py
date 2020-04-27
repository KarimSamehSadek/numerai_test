#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd  
training_data = pd.read_csv('numerai_training_data.csv')


# In[16]:


from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(training_data.iloc[:,0:21], training_data['target'], test_size=0.3, random_state=0)


# In[18]:


from sklearn.svm import SVC as svc
clf = svc(C=1.0).fit(features_train, labels_train)


# In[19]:


predictions = clf.predict(features_test)


# In[20]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(predictions,labels_test)


# In[21]:


accuracy


# In[ ]:




