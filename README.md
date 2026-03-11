# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import dataset and print head,info of the dataset
2.Import kmeans and fit it to the dataset 
3.Plot the graph using elbow method and predict the predicted array 
4.Plot the customer segments
```

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Anikha Pillai 
RegisterNumber: 25009524  
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
dataset = pd.read_csv('/mnt/data/Mall_Customers.csv')
print(dataset.head())
X = dataset.iloc[:, [3, 4]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

print("WCSS Values:")
print(wcss)


plt.figure()
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
print("Cluster Labels:")
print(y_kmeans)
plt.figure()
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50)
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50)
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50)
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50)
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            s=200, marker='X')


```

## Output:
<img width="1033" height="700" alt="image" src="https://github.com/user-attachments/assets/d6b7bf90-d69c-4514-bfdb-b9f030d01f2b" />
<img width="987" height="216" alt="image" src="https://github.com/user-attachments/assets/5ac0ba8b-dba8-4c26-b76c-7f648b6bb9a4" />
<img width="951" height="691" alt="image" src="https://github.com/user-attachments/assets/c39506cd-4e44-40b8-bbb9-110a36328b66" />





## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
