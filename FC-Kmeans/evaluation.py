
import os

os.environ['OMP_NUM_THREADS'] = '1'
import csv
import numpy as np
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
import os
from matplotlib import pyplot
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from main import K_Means
from main import FC_K_Means

def SSE(data, centroids, labels):
    sse = 0
    for i in range(len(data)):
        centroid = centroids[labels[i]]
        squared_distance = np.sum((data[i] - centroid) ** 2)
        sse += squared_distance
    return sse

# sample data uniformly
def uniform_sampling(data, n_samples):
    n_total_samples = data.shape[0]
    indices = np.linspace(0, n_total_samples - 1, n_total_samples, dtype=int)
    selected_indices = np.sort(np.random.choice(indices, size=n_samples, replace=False))
    selected_samples = data[selected_indices]
    return selected_samples

def uniform_sampling_indices(data, n_samples):
    n_total_samples = data.shape[0]
    indices = np.linspace(0, n_total_samples - 1, n_total_samples, dtype=int)
    selected_indices = np.sort(np.random.choice(indices, size=n_samples, replace=False))
    selected_samples = data[selected_indices]
    return selected_indices

# choose fix point according to density
def fix_point_choosing(data, n_points):
    # Compute the density of each point using KDE
    kde = KernelDensity(bandwidth=0.5)
    densities = np.exp(kde.fit(data).score_samples(data))

    # Normalize the density values
    densities /= np.sum(densities)

    # Compute the cumulative distribution function
    cdf = np.cumsum(densities)

    # Choose points randomly according to their density
    chosen_indices = []
    while len(chosen_indices) < n_points:
        random_number = np.random.rand()
        index = np.searchsorted(cdf, random_number)
        if index not in chosen_indices:
            chosen_indices.append(index)

    chosen_points = data[chosen_indices]

    return chosen_points

# import dataset
data = np.loadtxt(open("uber_raw_data_jun14.csv","rb"),delimiter=",",skiprows=1,usecols=[1,2])
#randomly sample data
data=uniform_sampling(data,60000)
#sample 6000 point for SI evaluation
sample_index=uniform_sampling_indices(data,6000)
#randomly choose 5 fix points
fix_point=fix_point_choosing(data,n_points=5)
#point=[[40.6521,-73.9616],[40.7543,-73.9702],[40.7493,-73.9727],[ 40.7597, -73.9915],[40.7196,-73.9547]]
#fix_point=np.array(point)
#plot 60000 data with 5 fix points


#######Test for fix point =3
independent_run=30
fixpointnum=3
k=np.array([3,4,5,6,7,8,9,10,15,20,50])
#Three evalutations
SE=np.zeros((len(k),4))
DB=np.zeros((len(k),4))
SI=np.zeros((len(k),4))

for i in np.arange(len(k)):
    #30 iterationprint(i,'step')、、
    print('k=',k[i],'step')
    inner_SE=np.zeros((independent_run,4))
    inner_DB=np.zeros((independent_run,4))
    inner_SI=np.zeros((independent_run,4))
    for j in np.arange(independent_run):
        print(j,'step')
        #initialize model
        kmeans=KMeans(n_clusters=k[i],n_init=1,init='random').fit(data)
        kmeanspp=KMeans(n_clusters=k[i],n_init=1).fit(data)
        FC_kmeans1=FC_K_Means(k=k[i],f=fixpointnum,function_type='FC')
        FC_kmeans2=FC_K_Means(k=k[i],f=fixpointnum,function_type='FC2')
        #fit data
        FC_kmeans1.fit(data,fix_point[:3])
        FC_kmeans2.fit(data,fix_point[:3])
        sse=np.array([kmeans.inertia_,kmeanspp.inertia_,SSE(data,FC_kmeans1.centroid,FC_kmeans1.label),SSE(data,FC_kmeans2.centroid,FC_kmeans2.label)])
        db=np.array([davies_bouldin_score(data,kmeans.labels_),davies_bouldin_score(data,kmeanspp.labels_),davies_bouldin_score(data,FC_kmeans1.label),davies_bouldin_score(data,FC_kmeans2.label)])
        #print(i,'db')
        si=np.array([silhouette_score(data[sample_index],kmeans.labels_[sample_index]),silhouette_score(data[sample_index],kmeanspp.labels_[sample_index]),
                     silhouette_score(data[sample_index],FC_kmeans1.label[sample_index]),silhouette_score(data[sample_index],FC_kmeans2.label[sample_index])])
       # print(i,'si')        
        inner_SE[j]=sse
        inner_DB[j]=db
        inner_SI[j]=si
        ###
        
    SE[i]=np.mean(inner_SE,axis=0)
    DB[i]=np.mean(inner_DB,axis=0)
    SI[i]=np.mean(inner_SI,axis=0)
    print(SE[i])
    print(DB[i])
    print(SI[i])
####save data
with open("Fixpoin=3.csv", "w",newline='') as f:
    csv_writer = csv.writer(f)
    # The first row must be "Id, Category"
    row=['','','SE','','','','','DB','','','','','Silhoutte','','']
    csv_writer.writerow(row)
    row=['number of cluster','Kmeans','Kmeans++','FC Kmeans','FC Kmeans2','','Kmeans','Kmeans++','FC Kmeans','FC Kmeans2','','Kmeans','Kmeans++','FC Kmeans','FC Kmeans2']
    csv_writer.writerow(row)
    # For the rest of the rows, each image id corresponds to a predicted class.
    for i in range(SE.shape[0]):
        row=[k[i],SE[i,0],SE[i,1],SE[i,2],SE[i,3],'',DB[i,0],DB[i,1],DB[i,2],DB[i,3],'',SI[i,0],SI[i,1],SI[i,2],SI[i,3]]
        csv_writer.writerow(row)
#        
#       
#        
#####Test for fix point =4
fixpointnum=4
k=np.array([4,5,7,8,9,11,12,13,20,27,67])
#Three evalutations
SE_B=np.zeros((len(k),4))
DB_B=np.zeros((len(k),4))
SI_B=np.zeros((len(k),4))

for i in np.arange(len(k)):
    #30 iteration
    print('k=',k[i],'step')
    inner_SE=np.zeros((independent_run,4))
    inner_DB=np.zeros((independent_run,4))
    inner_SI=np.zeros((independent_run,4))
    for j in np.arange(independent_run):
        print(j,'step')
        #initialize model
        kmeans=KMeans(n_clusters=k[i],n_init=1,init='random').fit(data)
        kmeanspp=KMeans(n_clusters=k[i],n_init=1).fit(data)
        FC_kmeans1=FC_K_Means(k=k[i],f=fixpointnum,function_type='FC')
        FC_kmeans2=FC_K_Means(k=k[i],f=fixpointnum,function_type='FC2')
        #fit data
        FC_kmeans1.fit(data,fix_point[:4])
        FC_kmeans2.fit(data,fix_point[:4])
        sse=np.array([kmeans.inertia_,kmeanspp.inertia_,SSE(data,FC_kmeans1.centroid,FC_kmeans1.label),SSE(data,FC_kmeans2.centroid,FC_kmeans2.label)])
       # print(i,'sse')
        db=np.array([davies_bouldin_score(data,kmeans.labels_),davies_bouldin_score(data,kmeanspp.labels_),davies_bouldin_score(data,FC_kmeans1.label),davies_bouldin_score(data,FC_kmeans2.label)])
        #print(i,'db')
        si=np.array([silhouette_score(data[sample_index],kmeans.labels_[sample_index]),silhouette_score(data[sample_index],kmeanspp.labels_[sample_index]),
                     silhouette_score(data[sample_index],FC_kmeans1.label[sample_index]),silhouette_score(data[sample_index],FC_kmeans2.label[sample_index])])
        #print(i,'si') 
        inner_SE[j]=sse
        inner_DB[j]=db
        inner_SI[j]=si
        
    SE_B[i]=np.mean(inner_SE,axis=0)
    DB_B[i]=np.mean(inner_DB,axis=0)
    SI_B[i]=np.mean(inner_SI,axis=0)
    print(i,'step')
    print(SE_B[i])
    print(DB_B[i])
    print(SI_B[i])
####save data
with open("Fixpoin=4.csv", "w",newline='') as f:
    csv_writer = csv.writer(f)
    # The first row must be "Id, Category"
    row=['','','SE','','','','','DB','','','','','Silhoutte','','']
    csv_writer.writerow(row)
    row=['number of cluster','Kmeans','Kmeans++','FC Kmeans','FC Kmeans2','','Kmeans','Kmeans++','FC Kmeans','FC Kmeans2','','Kmeans','Kmeans++','FC Kmeans','FC Kmeans2']
    csv_writer.writerow(row)
    # For the rest of the rows, each image id corresponds to a predicted class.
    for i in range(SE_B.shape[0]):
        row=[k[i],SE_B[i,0],SE_B[i,1],SE_B[i,2],SE_B[i,3],'',DB_B[i,0],DB_B[i,1],DB_B[i,2],DB_B[i,3],'',SI_B[i,0],SI_B[i,1],SI_B[i,2],SI_B[i,3]]
        csv_writer.writerow(row)

        
#####Test for fix point=5
fixpointnum=5
k=np.array([5,7,8,10,12,13,15,17,25,33,83])
#Three evalutations
SE_C=np.zeros((len(k),4))
DB_C=np.zeros((len(k),4))
SI_C=np.zeros((len(k),4))

for i in np.arange(len(k)):
    #30 iterationp
    print('k=',k[i],'step')
    inner_SE=np.zeros((independent_run,4))
    inner_DB=np.zeros((independent_run,4))
    inner_SI=np.zeros((independent_run,4))
    for j in np.arange(independent_run):
        print(j,'step')
        #initialize model
        kmeans=KMeans(n_clusters=k[i],n_init=1,init='random').fit(data)
        kmeanspp=KMeans(n_clusters=k[i],n_init=1).fit(data)
        FC_kmeans1=FC_K_Means(k=k[i],f=fixpointnum,function_type='FC')
        FC_kmeans2=FC_K_Means(k=k[i],f=fixpointnum,function_type='FC2')
        #fit data
        FC_kmeans1.fit(data,fix_point[:5])
        FC_kmeans2.fit(data,fix_point[:5])
        sse=np.array([kmeans.inertia_,kmeanspp.inertia_,SSE(data,FC_kmeans1.centroid,FC_kmeans1.label),SSE(data,FC_kmeans2.centroid,FC_kmeans2.label)])
        #print(i,'sse')
        db=np.array([davies_bouldin_score(data,kmeans.labels_),davies_bouldin_score(data,kmeanspp.labels_),davies_bouldin_score(data,FC_kmeans1.label),davies_bouldin_score(data,FC_kmeans2.label)])
        #print(i,'db')
        si=np.array([silhouette_score(data[sample_index],kmeans.labels_[sample_index]),silhouette_score(data[sample_index],kmeanspp.labels_[sample_index]),
                     silhouette_score(data[sample_index],FC_kmeans1.label[sample_index]),silhouette_score(data[sample_index],FC_kmeans2.label[sample_index])])
        #print(i,'si') 
        inner_SE[j]=sse
        inner_DB[j]=db
        inner_SI[j]=si
        
    SE_C[i]=np.mean(inner_SE,axis=0)
    DB_C[i]=np.mean(inner_DB,axis=0)
    SI_C[i]=np.mean(inner_SI,axis=0)
    print(i,'step')
    print(SE_C[i])
    print(DB_C[i])
    print(SI_C[i])   
###save data    
with open("Fixpoin=5.csv", "w",newline='') as f:
    csv_writer = csv.writer(f)
    # The first row must be "Id, Category"
    row=['','','SE','','','','','DB','','','','','Silhoutte','','']
    csv_writer.writerow(row)
    row=['number of cluster','Kmeans','Kmeans++','FC Kmeans','FC Kmeans2','','Kmeans','Kmeans++','FC Kmeans','FC Kmeans2','','Kmeans','Kmeans++','FC Kmeans','FC Kmeans2']
    csv_writer.writerow(row)
    # For the rest of the rows, each image id corresponds to a predicted class.
    for i in range(SE_C.shape[0]):
        row=[k[i],SE_C[i,0],SE_C[i,1],SE_C[i,2],SE_C[i,3],'',DB_C[i,0],DB_C[i,1],DB_C[i,2],DB_C[i,3],'',SI_C[i,0],SI_C[i,1],SI_C[i,2],SI_C[i,3]]
        csv_writer.writerow(row)
#
#
#
#
#
##draw picture and save picture
##fit data
kmeans=KMeans(n_clusters=8,n_init=1,init='random').fit(data)
kmeanspp=KMeans(n_clusters=8,n_init=1).fit(data)
FC_kmeans1=FC_K_Means(k=8,f=3,function_type='FC')
FC_kmeans2=FC_K_Means(k=8,f=3,function_type='FC2')
FC_kmeans1.fit(data,fix_point[:3])
FC_kmeans2.fit(data,fix_point[:3])

####Picture of Dataset#####
pyplot.scatter(data[:,0],data[:,1],s=0.5)
pyplot.scatter(fix_point[:3,0],fix_point[:3,1],s=100, marker='*',c='r',label='Fixed point')
pyplot.legend()
pyplot.title("Data set with randomly selected fixed points")
pyplot.savefig('Dataset.png')
pyplot.show()

######Picture of FC-Kmeans#######
pyplot.scatter(data[:,0], data[:,1], c=FC_kmeans1.label,s=0.5)
pyplot.scatter(FC_kmeans1.centroid[:(FC_kmeans1.k-FC_kmeans1.f),0], FC_kmeans1.centroid[:(FC_kmeans1.k-FC_kmeans1.f),1], marker='.',c='r', s=100,label='Non-fixed point')
pyplot.scatter(FC_kmeans1.centroid[(FC_kmeans1.k-FC_kmeans1.f):,0], FC_kmeans1.centroid[(FC_kmeans1.k-FC_kmeans1.f):,1], marker='*',c='r', s=100,label='Fixed point')
pyplot.legend()
pyplot.title("FC-Kmeans(k=8)")
pyplot.savefig('FC_Kmeans.png')
pyplot.show()
######Picture of FC-Kmeans2#######
pyplot.scatter(data[:,0], data[:,1], c=FC_kmeans2.label,s=0.5)
pyplot.scatter(FC_kmeans2.centroid[:(FC_kmeans2.k-FC_kmeans2.f),0], FC_kmeans2.centroid[:(FC_kmeans2.k-FC_kmeans2.f),1], marker='.',c='r', s=100,label='Non-fixed point')
pyplot.scatter(FC_kmeans2.centroid[(FC_kmeans2.k-FC_kmeans2.f):,0], FC_kmeans2.centroid[(FC_kmeans2.k-FC_kmeans2.f):,1], marker='*',c='r', s=100,label='Fixed point')
pyplot.legend()
pyplot.title("FC-Kmeans2(k=8)")
pyplot.savefig('FC_Kmeans2.png')
pyplot.show()
#####Picture of Kmeans++######
pyplot.scatter(data[:,0], data[:,1], c=kmeanspp.labels_,s=0.5)
pyplot.scatter(kmeanspp.cluster_centers_[:,0], kmeanspp.cluster_centers_[:,1], marker='.',c='r', s=100,label='Non-fixed point')
pyplot.legend()
pyplot.title("Kmeans++(k=8)")
pyplot.savefig('Kmeans++.png')
pyplot.show()
