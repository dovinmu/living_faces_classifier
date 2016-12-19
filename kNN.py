'''
Some code cribbed from http://blog.yhat.com/posts/classification-using-knn-and-python.html

Filtering for faces with 10 vectors or more, on k=1 I get a success rate of 96.7%.
That's okay for a first try, but I'm skipping over 5000 other people. I need to
factor in distance between vectors to decide if a vector should be classified
according to its nearest neighbors or 'unknown'.


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

def confidenceAccuracy(erroneous_confidence, correct_confidence):
    return 1 - erroneous_confidence / (erroneous_confidence + correct_confidence)

def loadDF(skipnum=10):
    os.chdir('/home/rowan/Documents/datasets/labeled_faces_of_the_wild/lfw_data/lfw_vectors')
    vec_list = []
    skipcount = 0
    for folder in sorted(os.listdir()):
        os.chdir(folder)
        if len(os.listdir()) < 1: # this is one way of making the model more accurate...
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
    return df

subset_idx = np.random.uniform(0,1,len(df)) <= 1
df2 = df[subset_idx]
print('taking a subset of size {}'.format(len(df2)))

test_idx = np.random.uniform(0,1,len(df2)) <= 0.3
train = df2[test_idx==False]
test = df2[test_idx==True]
print('train dataset size: {}, test dataset size: {}'.format(len(train), len(test)))

response = 'name'
features = [col for col in df2.columns if col not in [response]]
uniques = train[response].unique()
results = []
for k in range(8,10,1):
    neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
    print('\n\ntraining model on k={}'.format(k))

    start_time = time.time()
    neigh.fit(train[features], train[response])
    print('took {} minutes'.format( round((time.time()-start_time)/60, 2) ))
    '''
    print('getting predictions')
    start_time = time.time()
    preds = neigh.predict(test[features])
    percentCorrect = np.where(preds==test[response],1,0).sum() / float(len(test))
    print('\t\tNeighbors: %d, Percent Correct: %3f' % (k, percentCorrect) )
    print('took {} minutes'.format( round((time.time()-start_time)/60, 2) ))
    '''
    start_time = time.time()
    print('getting prbabilistic predictions')
    predictions_probabilistic = neigh.predict_proba(test[features])
    print('took {} minutes'.format( round((time.time()-start_time)/60, 2) ))
    high_confidence_and_wrong_count = 0
    total_erroneous_confidence = 0
    total_correct_confidence = 0

    for prediction_idx in range(len(predictions_probabilistic)):
        prediction = predictions_probabilistic[prediction_idx]
        actual_name = test[response].iloc[prediction_idx]
        most_probable_idx = prediction.argmax()
        predicted_name = uniques[most_probable_idx]
        if actual_name != predicted_name:
            total_erroneous_confidence += prediction.max()
            if prediction.max() > 0.5:
                print('confidence: {:6}   actual: {:28} predicted: {:20}'.format( round(prediction.max(),3), actual_name, predicted_name ))
                high_confidence_and_wrong_count += 1
        else:
            total_correct_confidence += prediction.max()

    print('total of confidence in wrong guesses: {}\ntotal of confidence in correct guesses: {}\nconfidence-corrected accuracy: {}'.format(total_erroneous_confidence, total_correct_confidence, round(confidenceAccuracy(total_erroneous_confidence,total_correct_confidence),4) ))
    results.append([k, '0.x', high_confidence_and_wrong_count, total_erroneous_confidence, total_correct_confidence])

formatter = "{:2}   {:15}   {:20}   {:20}  {:20}  {:20}"
print(formatter.format( 'k', 'percent correct', 'wrong w/ high conf.', 'err confidence', 'correct confidence', 'confidence-corrected accuracy' ))
for row in results:
    print(formatter.format( row[0], row[1], row[2], round(row[3],3), round(row[4],3), round(confidenceAccuracy(row[3],row[4]), 3) ))
