import numpy as np
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
import os
from matplotlib import pyplot
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from main import K_Means

# import dataset
data = np.loadtxt(open("uber_raw_data_jun14.csv","rb"),delimiter=",",skiprows=63845,usecols=[1,2])
pyplot.scatter(data[:,0],data[:,1],s=0.5)
pyplot.show()


def SSE(data, centroids, labels):
    sse = 0
    for i in range(len(data)):
        centroid = centroids[labels[i]]
        squared_distance = np.sum((data[i] - centroid) ** 2)
        sse += squared_distance
    return sse

# 貌似没有用到
def select_nfixpoint(data, n):
    # Create a KernelDensity object and fit the data
    kde = KernelDensity(bandwidth=0.5)
    kde.fit(data)

    # Calculate the logarithm of the estimated density values for each data point
    log_densities = kde.score_samples(data)

    # Sort the data points based on their density values in descending order
    sorted_indices = np.argsort(log_densities)[::-1]

    selected_indices = sorted_indices[:n]

    # Get the selected data points
    selected_data = data[selected_indices]

    # Set a minimum distance threshold between selected points
    min_distance = 0.2

    # Remove any selected points that are closer than the minimum distance
    final_selected_indices = [selected_indices[0]]
    for i in range(1, len(selected_indices)):
        distances = cdist(selected_data[i].reshape(1, -1), data[final_selected_indices])
        if np.min(distances) > min_distance:
            final_selected_indices.append(selected_indices[i])

    # Get the final selected data points
    final_selected_data = data[final_selected_indices]
    return final_selected_indices


# 自己写的kmeans
k_meanspp = K_Means(k=3,init='kmeanpp')
k_meanspp.fit(data)
center=np.arange(k_meanspp.centroid.shape[0])
pyplot.scatter(data[:,0], data[:,1], c=k_meanspp.label)
pyplot.scatter(k_meanspp.centroid[:,0], k_meanspp.centroid[:,1], marker='*',c='r', s=150)

pyplot.show()

# 调用的kmeans
n_clusters=3
cluster = KMeans(n_clusters=n_clusters,n_init=1).fit(data)
centroid=cluster.cluster_centers_
y_pred = cluster.labels_#获取训练后对象的每个样本的标签
centtrod = cluster.cluster_centers_
pyplot.scatter(data[:,0], data[:,1], c=y_pred)
pyplot.scatter(centtrod[:,0], centtrod[:,1], marker='*',c='r', s=150)

pyplot.show()

## FC-Kmeans

##

print(SSE(data,k_meanspp.centroid,k_meanspp.label))
print(SSE(data,centtrod,y_pred))
print(cluster.inertia_)
