'''
Some code cribbed from http://blog.yhat.com/posts/classification-using-knn-and-python.html
'''

import os
import time
import random
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

os.chdir('/home/rowan/Documents/datasets/labeled_faces_of_the_wild/lfw_data/lfw_vectors')
vec_list = []
skipcount = 0
for folder in sorted(os.listdir()):
    os.chdir(folder)
    if len(os.listdir()) < 20: # this is one way of making the model more accurate...
        os.chdir('..')
        skipcount += 1
        continue
    print(folder)
    for npy in os.listdir():
        vec = np.load(npy)
        vec = vec/np.linalg.norm(vec)
        vec_list.append([folder, *vec])
    os.chdir('..')
print('='*50, '\nloading {} vectors into a dataframe'.format(len(vec_list)))
if skipcount:
    print('skipped {} of {} people'.format( skipcount, len(os.listdir()) ))
df = DataFrame(vec_list,columns=['name', *([i for i in range(4096)])])

subset_idx = np.random.uniform(0,1,len(df)) <= 1
df2 = df[subset_idx]
print('taking a subset of size {}'.format(len(df2)))

test_idx = np.random.uniform(0,1,len(df2)) <= 0.3
train = df2[test_idx==False]
test = df2[test_idx==True]
print('train dataset size: {}, test dataset size: {}'.format(len(train), len(test)))

response = 'name'
features = [col for col in df2.columns if col not in [response]]
numericPrediction = False

results = []
for k in range(1,3,1):
    # fit the knn algorithm to the test data
    neigh = KNeighborsClassifier(n_neighbors=k)
    print('training model on k={}'.format(k))
    start_time = time.time()
    neigh.fit(train[features], train[response])
    print('took {} minutes'.format( round((time.time()-start_time)/60, 2) ))
    print('getting predictions')
    start_time = time.time()
    preds = neigh.predict(test[features])
    print('took {} minutes'.format( round((time.time()-start_time)/60,2) ))
    percentCorrect = np.where(preds==test[response],1,0).sum() / float(len(test))
    if numericPrediction:
        averageError = abs(preds-list(test[response])).sum() / float(len(test))
        print('\t\tNeighbors: %d, Percent Correct: %3f, Average Error: %2f' % (k, percentCorrect, averageError) )
        results.append([k, percentCorrect, averageError])
    else:
        print('\t\tNeighbors: %d, Percent Correct: %3f' % (k, percentCorrect) )
        results.append([k, percentCorrect])
