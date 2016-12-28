import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans,vq
from sklearn.cluster import KMeans
from sklearn import metrics
import model_utils


def homogeneity(codex, df):
    '''
    What does it mean to gauge accuracy for k-means?
    Well, k-means is trying for each cluster to exactly correspond to the set of
    all elements with that label / or things that actually belong in that set.

    So homogeneous clusters == good, heterogeneous == bad.
    I could define the homogeneity of a cluster to be (num of mode) / (total size of cluster),
    so a perfectly homogeneous cluster would have homogeneity=1.0.
    And the homogeneity of all the clusters would be sum(num of modes) / (total size).

    The problem with this definition is that it will be maximized when each point
    is in its own cluster. So let's only count the cluster with the most occurances
    of the name (as long as that name is the mode for the cluster).
    '''

    name_to_mode_count_list = {}
    for cluster_id in range(n_clusters):
        x = df[codex==cluster_id]
        try:
            most_predicted_name = x['name'].mode()[0]
        except:
            if len(x) == 0:
                #print('no points in this cluster')
                continue
            most_predicted_name = x['name'].iloc[0]
        add_to_total = len(x[x['name']==most_predicted_name])
        #print(most_predicted_name, add_to_total, len(x))
        if most_predicted_name not in name_to_mode_count_list:
            name_to_mode_count_list[most_predicted_name] = []
        name_to_mode_count_list[most_predicted_name].append(add_to_total)
    mode_count = 0
    for name in name_to_mode_count_list.keys():
        mode_count += max(name_to_mode_count_list[name])
        #print(name, max(name_to_mode_count_list[name]))
    return  round(mode_count/len(df),3)

sers = {}

#2, 4, 6, 8, 10, 20, 30,
for threshold in [2, 4, 6, 8, 10, 20, 30, 40, 50]:
    if threshold in sers:
        continue
    print('computing with threshold={}'.format(threshold))
    df = model_utils.loadDF(data_threshold=threshold)
    df2 = df.drop(df.columns[0], axis=1)
    results = {}
    for n_clusters in range(100,801,25):

        print('\ncomputing sklearn kmeans+metrics on {} clusters'.format(n_clusters))
        start_time = time.time()
        model = KMeans(n_clusters=n_clusters).fit(df2.values)
        labels = model.labels_
        silhouette_coeff = metrics.silhouette_score(df2, labels, metric='euclidean')
        #homog = homogeneity(labels, df)

        print('took {} minutes'.format( round((time.time()-start_time)/60,2) ))

        print("\t\t\t\t avg. silhouette coefficient: {}".format(
            silhouette_coeff
        ))
        results[n_clusters] = silhouette_coeff#[homog, silhouette_coeff]
    ser = pd.Series(results)
    sers[threshold] = ser

for threshold in sorted(sers.keys()):
    ser = sers[threshold]
    ser.plot(label='>='+str(threshold))

plt.style.use('ggplot')
plt.title('K-Means silhouette coefficient by data sample threshold')
plt.ylabel('Silhouette coefficient')
plt.xlabel('number of clusters')
plt.legend()
plt.show()
