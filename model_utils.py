import os
from pandas import DataFrame
import numpy as np

def loadDF(data_threshold=10):
    try:
        with open('dataset_dirname') as f:
            dirname = f.read().strip()
    except:
        raise Exception('Could not find location of LFW dataset.')

    working_dir = os.getcwd()
    os.chdir(dirname)
    vec_list = []
    skipcount = 0
    for folder in sorted(os.listdir()):
        os.chdir(folder)
        if len(os.listdir()) < data_threshold:
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
        print('loaded {} of {} people'.format( len(os.listdir())-skipcount, len(os.listdir()) ))
    df = DataFrame(vec_list,columns=['name', *([i for i in range(4096)])])
    os.chdir(working_dir)
    return df
