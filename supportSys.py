import pandas as pd
import numpy as np
import math

# Importing all Classified Data
pdataset = pd.read_csv('positive_tweets.csv', names=['tweet', 'classified'])
ndataset = pd.read_csv('negative_tweets.csv', names=['tweet', 'classified'])
nudataset = pd.read_csv('neutral_tweets.csv', names=['tweet', 'classified'])

# Class for Cleaning unusual New Lines and NaN values
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
import SL_NB_Processor as cnp
machine, X, y = cnp.processor(mainset)
prediction = cnp.prediction(machine, X, y)
print("Supervised Learning: Naive Bayes \n Results:")
print(prediction)

# UN-SUPERVISED LEARNING ALGORITHM
import USL_NB_Processor as cnp
machine, X, y = cnp.processor(mainset)
# No Train Test Split and hence no need of y
del y
pred_df = cnp.prediction(machine, X)
print("Un-Supervised Learning: Naive Bayes \n Results:")
print(pred_df.head())
pred_df.to_csv('NBpredictions')

# TODO Random Forest / Decision Tree SL & USL