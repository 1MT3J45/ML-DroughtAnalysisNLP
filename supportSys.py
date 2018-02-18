import pandas as pd
import numpy as np
import math

pdataset = pd.read_csv('positive_tweets.csv', names=['tweet', 'classified'])
ndataset = pd.read_csv('negative_tweets.csv', names=['tweet', 'classified'])
nudataset = pd.read_csv('neutral_tweets.csv', names=['tweet', 'classified'])

def na_remover(df):
    # Cleaning up the NaNs
    array1 = df.iloc[:, 0].dropna().reset_index()
    array2 = df.iloc[:, 1].dropna().reset_index()
    print("Cleaning done")
    # Realign with correct Indexes
    array1 = array1.iloc[:, 1]
    array2 = array2.iloc[:, 1]
    print("Realigned values")
    df = pd.concat([array1, array2], axis=1)
    print("Dataframe is now NaN free!")
    return df

pos_df = na_remover(pdataset)
neg_df = na_remover(ndataset)

# Merging Positives and Negatives for Analysis
mainset = pd.concat([pos_df, neg_df], axis=0, ignore_index=True)
mainset['classified'] = mainset['classified'].map({'Positive':1, 'Negative':0})

# SUPERVISED LEARNING ALGORITHM