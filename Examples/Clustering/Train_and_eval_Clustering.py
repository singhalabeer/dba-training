# Databricks notebook source
# MAGIC %md
# MAGIC Clustering - Introduction
# MAGIC In contrast to supervised machine learning, unsupervised learning is used when there is no "ground truth" from which to train and validate label predictions. The most common form of unsupervised learning is clustering, which is simllar conceptually to classification, except that the the training data does not include known values for the class label to be predicted. Clustering works by separating the training cases based on similarities that can be determined from their feature values. Think of it this way; the numeric features of a given entity can be thought of as vector coordinates that define the entity's position in n-dimensional space. What a clustering model seeks to do is to identify groups, or clusters, of entities that are close to one another while being separated from other clusters.
# MAGIC 
# MAGIC For example, let's take a look at a dataset that contains measurements of different species of wheat seed.

# COMMAND ----------

import pandas as pd

# load the training dataset asml xshift
#!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/seeds_tutorial.csv
data = pd.read_csv('arf1_seed_full.csv')


# Display a random sample of 10 observations (just the features)
features = data[data.columns[0:1]]
features.sample(10)

#seed_species = data['SPECIES']
#seed_species.values

#seed_overlays = data['OVERLAY']
#seed_overlays.values


# COMMAND ----------

# MAGIC %md
# MAGIC As you can see, the dataset contains six data points (or features) for each instance (observation) of a seed. So you could interpret these as coordinates that describe each instance's location in six-dimensional space.
# MAGIC 
# MAGIC Now, of course six-dimensional space is difficult to visualise in a three-dimensional world, or on a two-dimensional plot; so we'll take advantage of a mathematical technique called Principal Component Analysis (PCA) to analyze the relationships between the features and summarize each observation as coordinates for two principal components - in other words, we'll translate the six-dimensional feature values into two-dimensional coordinates.

# COMMAND ----------

features.values

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Normalize the numeric features so they're on the same scale
scaled_features = MinMaxScaler().fit_transform(features[data.columns[0:2]])

# Get two principal components
pca = PCA(n_components=2).fit(scaled_features)
features_2d = pca.transform(scaled_features)
features_2d[0:10]

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have the data points translated to two dimensions, we can visualize them in a plot:

# COMMAND ----------

import matplotlib.pyplot as plt

%matplotlib inline

plt.scatter(feature[:,0],feature[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Data')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Hopefully you can see at least two, arguably three, reasonably distinct groups of data points; but here lies one of the fundamental problems with clustering - without known class labels, how do you know how many clusters to separate your data into?
# MAGIC 
# MAGIC One way we can try to find out is to use a data sample to create a series of clustering models with an incrementing number of clusters, and measure how tightly the data points are grouped within each cluster. A metric often used to measure this tightness is the within cluster sum of squares (WCSS), with lower values meaning that the data points are closer. You can then plot the WCSS for each model.

# COMMAND ----------

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
%matplotlib inline


# Create 10 models with 1 to 10 clusters
wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters = i)
    # Fit the data points
    kmeans.fit(features.values)
    # Get the WCSS (inertia) value
    wcss.append(kmeans.inertia_)

#Plot the WCSS values onto a line graph
print(wcss)

#Plot the WCSS values onto a line graph
plt.plot(range(2,11), wcss)
plt.title('WCSS by Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# COMMAND ----------

from sklearn.cluster import KMeans

# Create a model based on 3 centroids
model = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=10000)
# Fit to the data and predict the cluster assignments for each data point
km_clusters = model.fit_predict(features.values)
# View the cluster assignments
km_clusters


# COMMAND ----------

import matplotlib.pyplot as plt
%matplotlib inline


def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        #print(samples[int(sample)])
        plt.scatter(int(sample),samples[sample], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

#features
    
plot_clusters(features.values, km_clusters)

# COMMAND ----------

import matplotlib.pyplot as plt
%matplotlib inline


def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange', 3:'cyan'}
    mrk_dic = {0:'*',1:'x',2:'+', 3:'.'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        #print(samples[int(sample)])
        plt.scatter(int(sample),samples[int(sample)], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

#features
    
plot_clusters(features.values, km_clusters)

# COMMAND ----------

import matplotlib.pyplot as plt
%matplotlib inline

def plot_clusters_v2(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(int(sample),samples[int(sample)], color = colors[sample], marker=markers[sample], s=100)
        #plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()

#seed_species = data[data.columns[7]]
plot_clusters_v2(features.values, seed_species.values)

# COMMAND ----------

seed_overlays.values

# COMMAND ----------

seed_species.values

# COMMAND ----------

from sklearn.metrics import confusion_matrix
y_true = seed_species.values
y_pred =  km_clusters
confusion_matrix(y_true, y_pred)

# COMMAND ----------

from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# COMMAND ----------

from sklearn.metrics import confusion_matrix
y_true = seed_overlays.values
y_pred =  km_clusters
confusion_matrix(y_true, y_pred)

# COMMAND ----------

from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# COMMAND ----------


