import os
from pandas import DataFrame
import numpy as np

def loadDF(data_threshold=10, verbose=False):
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
        if verbose:
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

def print_stats_table():
    formatter = "\n{:10}\t{:15}\t{:15}"

    table = formatter.format( 'threshold', 'vectors', 'people' )
    for threshold in [1,2,4,6,8,10,20,30,40,50]:
        df = loadDF(data_threshold=threshold)
        table += formatter.format( threshold, len(df), len(df['name'].unique()) )

    print(table)
