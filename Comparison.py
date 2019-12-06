import sys
import matplotlib.pyplot as plt
from time import time
from sklearn.datasets import load_digits
digits = load_digits()

fig, axes = plt.subplots(2, 5, figsize=(10,5), subplot_kw={'xticks':(), 'yticks':()})
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)

# print len(digits.data) #1797
colors = ["#476A2A","#7851B8",'#BD3430','#4A2D4E','#875525',
          '#A83683','#4E655E','#853541','#3A3120','#535D8E']

t0=time()
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca = pca.fit(digits.data)
digits_pca = pca.transform(digits.data)

plt.figure(figsize=(10,10))
plt.xlim(digits_pca[:,0].min(), digits_pca[:,0].max())
plt.ylim(digits_pca[:,1].min(), digits_pca[:,1].max())

for i in range(len(digits.data)):
    plt.text(digits_pca[i,0], digits_pca[i,1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight':'bold', 'size':9})
plt.title('PCA')
plt.xlabel("first PC")
plt.ylabel("second PC")
print ("PCA time: ", time()-t0)
plt.show()

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(digits_pca)
scaled_p = scaler.transform(digits_pca)


#PCA -> DBSCAN
t2 = time()
from sklearn.cluster import DBSCAN
import numpy as np

db = DBSCAN(eps=0.122, min_samples=10).fit(scaled_p)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print ("number of clusters in pca-DBSCAN: ", n_clusters_)

plt.scatter(scaled_p[:,0], scaled_p[:,1], c=labels, s=60, edgecolors='black')
plt.title('PCA -> DBSCAN')
plt.xlabel("first PC")
plt.ylabel("second PC")
print ("DBSCAN time: ", time()-t2)
plt.show()



#PCA -> k-MEANS
t3 = time()
from sklearn.cluster import KMeans
km = KMeans(n_clusters=10, random_state=0)
labels_km = km.fit_predict(scaled_p)

plt.scatter(scaled_p[:,0], scaled_p[:,1], c=labels_km, s=60, edgecolors='black')
plt.title('PCA -> k-MEANS')
plt.xlabel("first PC")
plt.ylabel("second PC")
print("k-MEANS time: ", time())
plt.show()


print (labels_km)
print (labels)