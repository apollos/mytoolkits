import numpy as np
import os
import re
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt

input_path = '/home/yu/workspace/mytoolkits/python/nlp_demo/txtfile/vector'
output_path = '/home/yu/workspace/mytoolkits/python/nlp_demo/txtfile/result'

def generate_file_lst(in_path, postfix):
    file_list = []
    if os.path.isdir(in_path):
        files = os.listdir(in_path)
        for file_name in files:
            if os.path.isfile(os.path.join(in_path, file_name)):
                filename, fileext = os.path.splitext(file_name)
                if fileext == postfix:
                    file_list.append(os.path.join(in_path, file_name))
            elif os.path.isdir(os.path.join(in_path, file_name)):
                file_list += generate_file_lst(os.path.join(in_path, file_name))
    else:
        if os.path.isfile(in_path):
            filename, fileext = os.path.splitext(in_path)
            if fileext == postfix:
                file_list.append(in_path)
    return file_list

file_list = generate_file_lst(input_path, ".txt")
if len(file_list) == 0:
    print("Do not find bz2 file in %s" % input_path)
    exit(-1)

doc_cont_list = []
doc_indx_list = []
for vecfile in file_list:
    basename = os.path.basename(vecfile)
    '''Special action for my local case'''
    fname, fext = os.path.splitext(basename)
    fname, fext = os.path.splitext(fname)
    doc_num = re.sub(r'\D', "", fext)
    vec_cont = np.loadtxt(vecfile)
    doc_cont_list.append(vec_cont)
    doc_indx_list.append(doc_num)
print "Load All(%d) files" % len(doc_cont_list)

K = range(3, 100)
KM = [KMeans(n_clusters=k, max_iter=3000, init='k-means++', n_jobs=-1).fit(doc_cont_list) for k in K]
print "Complete kmeans search"
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(doc_cont_list, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D, axis=1) for D in D_k]
dist = [np.min(D, axis=1) for D in D_k]
avgWithinSS = [sum(d)/np.shape(doc_cont_list)[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(doc_cont_list)**2)/np.shape(doc_cont_list)[0]
bss = tss-wcss

kIdx = 10-1

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,
        markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')

