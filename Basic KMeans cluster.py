## Import the relevant libraries
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


## Load the data
raw_data = pd.read_csv('Countries-exercise.csv')
print(raw_data.hed())

# Remove the duplicate index column from the dataset.
data=raw_data.copy()
## Plot the data
# Plot the 'Longitude' and 'Latitude' columns
plt.scatter(data['Longitude'],data['Latitude'])
plt.xlim(-360,360)
plt.xlim(-180,180)
plt.show()
## Select the features
# Create a copy of that data and remove all parameters apart from Longitude and Latitude.
# Using Pandas Iloc fucntion
# iloc returns a Pandas Series when one row is selected, and a Pandas DataFrame when multiple rows are selected,
# or if any column in full is selected.
# To counter this, pass a single-valued list if you require DataFrame output.
x = data.iloc[:,1:3]
print(x)
# Clustering
# Here's the actual solution:
# Simply change kmeans = KMeans(2) to kmeans = KMeans(3) .
# Then run the remaining kernels until the end.

kmeans = KMeans(7)
kmeans.fit(x)
### Clustering Resutls
identifed_clusters = kmeans.fit_predict(x)
print(identifed_clusters)
# Creating a column 'clusters' in the data
data_wih_clusters = data.copy()
data_wih_clusters['Clusters']= identifed_clusters
print(data_wih_clusters)
 # Plot the data
# Plot the 'Longitude' and 'Latitude' columns to get final output graph
plt.scatter(data_wih_clusters['Longitude'],data_wih_clusters['Latitude'],c=data_wih_clusters['Clusters'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()
