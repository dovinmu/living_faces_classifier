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
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import model_utils

def confidenceAccuracy(erroneous_confidence, correct_confidence):
    return 1 - erroneous_confidence / (erroneous_confidence + correct_confidence)

sers = {}


for threshold in [1, 2, 4, 6, 8, 10, 20, 30, 40, 50]:
    if threshold in sers:
        continue
    df = model_utils.loadDF(data_threshold=threshold)
    print('loaded data with threshold={}'.format(threshold))

    if len(df) > 10000:
        sample_idx = np.random.uniform(0,1,len(df)) <= 0.3
        df = df[sample_idx==False]
        print('took sample of size {}'.format(len(df)))

    test_idx = np.random.uniform(0,1,len(df)) <= 0.3
    train = df[test_idx==False]
    test = df[test_idx==True]
    print('train dataset size: {}, test dataset size: {}'.format(len(train), len(test)))

    response = 'name'
    features = [col for col in df.columns if col not in [response]]
    uniques = train[response].unique()
    results = {}
    for k in range(1,20,1):
        neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
        print('\n\ntraining model on k={}'.format(k))

        start_time = time.time()
        neigh.fit(train[features], train[response])
        print('took {} seconds'.format( round((time.time()-start_time), 2) ))

        print('getting predictions')
        start_time = time.time()
        preds = neigh.predict(test[features])
        percentCorrect = np.where(preds==test[response],1,0).sum() / float(len(test))
        print('\t\tNeighbors: %d, Percent Correct: %3f' % (k, percentCorrect) )
        print('took {} minutes'.format( round((time.time()-start_time)/60, 2) ))

        print('percent correct: {}'.format( round(percentCorrect*100,2) ))
        results[k] = percentCorrect
        #results.append([k, percentCorrect])
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
        '''
    '''
    formatter = "{:2}   {:15}   {:20}   {:20}  {:20}  {:20}"
    print(formatter.format( 'k', 'percent correct', 'wrong w/ high conf.', 'err confidence', 'correct confidence', 'confidence-corrected accuracy' ))
    for row in results:
        print(formatter.format( row[0], row[1], row[2], round(row[3],3), round(row[4],3), round(confidenceAccuracy(row[3],row[4]), 3) ))
    '''
    ser = Series(results)
    sers[threshold] = ser

for threshold in sorted(sers.keys()):
    ser = sers[threshold]
    ser[:10].plot(label='>='+str(threshold))
plt.title("Percent correct by data sample threshold")
plt.xlabel("k")
plt.ylabel("percecnt correct")
plt.legend(loc=0)
plt.style.use('fivethirtyeight')
plt.show()
