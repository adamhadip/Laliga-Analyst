#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd 
import numpy as np 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler


# In[24]:


df = pd.read_csv('laliga_player_stats_english.csv')
df


# In[25]:


df.shape


# In[26]:


df.describe()


# In[27]:


df.info()


# In[28]:


from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df['Percentage of games played']= label_encoder.fit_transform(df['Percentage of games played'])
df['Percentage of full games played']= label_encoder.fit_transform(df['Percentage of full games played'])
df['Percentage of games started']= label_encoder.fit_transform(df['Percentage of games started'])
df['Percentage of games where substituted']= label_encoder.fit_transform(df['Percentage of games where substituted'])


# In[29]:


df.info()


# In[30]:


df.isnull().sum()


# In[31]:


df.dropna(inplace=True)
df


# In[32]:


df.isnull().sum()


# In[33]:


duplicate = df[df.duplicated()]
duplicate


# In[34]:


from claming import Cleansing, Matching


# In[35]:


clean = Cleansing()
match = Matching()


# In[36]:


match.levenshtein_match('Hodei Olega', 'Hodei Olega')


# In[37]:


groupby_Team = df.groupby(['Team'])[['Games played' , 'Shots' , 'Goals scored', 
                                     'Assists','Yellow Cards','Red Cards','Passes','Offsides','Penalties scored']].sum().reset_index()
groupby_Team


# In[38]:


from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'Team'.
groupby_Team['Team']= label_encoder.fit_transform(groupby_Team['Team'])


# In[39]:


groupby_Team.head()


# In[40]:


plt.figure(figsize=(15,10))
sns.heatmap(groupby_Team.corr(), cmap='magma', annot=True)
plt.show()


# In[41]:


groupby_Team.corr()


# In[42]:


plt.figure(figsize=(15,10))
plt.scatter(groupby_Team['Shots'], groupby_Team['Goals scored'])
plt.xlabel('Games Played')
plt.ylabel('Goal Scored')
plt.show()


# In[43]:


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(groupby_Team[['Shots','Goals scored']])
y_predicted


# In[44]:


groupby_Team.columns = ['Team','Games played' , 'Shots' , 'Goals scored', 
                                     'Assists','Yellow Cards','Red Cards','Passes','Offsides','Penalties scored']
groupby_Team['cluster']=y_predicted
groupby_Team


# In[51]:


data1 = groupby_Team[groupby_Team.cluster==0]
data2 = groupby_Team[groupby_Team.cluster==1]
data3 = groupby_Team[groupby_Team.cluster==2]

plt.scatter(groupby_Team['Shots'],groupby_Team['Goals scored'],color='green')
plt.scatter(groupby_Team['Shots'],groupby_Team['Goals scored'],color='red')
plt.scatter(groupby_Team['Shots'],groupby_Team['Goals scored'],color='yellow')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='black',marker='*',label='centroid')

plt.xlabel('Shots')
plt.ylabel('Goal Scored')
plt.legend()


# In[46]:


scaler = MinMaxScaler()
groupby_Team['Shots'] = scaler.fit_transform(groupby_Team[['Shots']])
groupby_Team['Goals scored'] = scaler.fit_transform(groupby_Team[['Goals scored']])


# In[47]:


bss = []
k_range = range(1,10)
for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++")
    km.fit(groupby_Team[['Games played','Goals scored']])
    bss.append(km.inertia_)

plt.xlabel('Cluster')
plt.ylabel('BSS')
plt.title('Find Elbow')
plt.plot(k_range,bss)


# In[48]:


kmeansmodel = KMeans(n_clusters= 3, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(groupby_Team)


# In[49]:


for i in range(2,12):
    labels= KMeans(n_clusters=i,init="k-means++",random_state=200).fit((groupby_Team[['Shots','Goals scored']])).labels_
    print ("Score for k(clusters) = "+str(i)+" is "
           +str(metrics.silhouette_score((groupby_Team[['Shots','Goals scored']]),labels,metric="euclidean",sample_size=500,random_state=200)))


# In[50]:


km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(groupby_Team[['Shots','Goals scored']])
groupby_Team['cluster']=y_predicted
groupby_Team


# In[ ]:




