import pandas as pd
import numpy as np

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
neu_df = na_remover(nudataset)

# Merging Positives and Negatives for Analysis
mainset = pd.concat([pos_df, neg_df, neu_df], axis=0, ignore_index=True)
mainset['classified'] = mainset['classified'].map({'Positive':1, 'Negative':0, 'Neutral':-1})

# SUPERVISED LEARNING ALGORITHM
import SL_NB_Processor as cnp
import plotter as pltr
machine, X, y = cnp.processor(mainset)
print("\nSupervised Learning: Naive Bayes \nResults:")
prediction = cnp.sl_prediction(machine, X, y)
print"SL: NBC - Conf. Matrix".center(45,'_'), "\n", prediction, "\n"
pltr.bars(prediction, plt_name="SL - Naive Bayes Classifier")

# UN-SUPERVISED LEARNING ALGORITHM
import USL_NB_Processor as cnp
machine, X, y = cnp.processor(mainset)
# No Train Test Split and hence no need of y
del y
pred_df = cnp.usl_prediction(machine, X)
print("Un-Supervised Learning: Naive Bayes \n Results:")
print"USL: NBC - Predictions".center(45,'_'), "\n", prediction, "\n"
print(pred_df.head())
pred_df.to_csv('NBpredictions')
gt = mainset['classified'].value_counts()
pr = pred_df.iloc[:, -1].value_counts()
pltr.biplt(gt, pr, "UnSupervised Naive Bayes")

# RFG
import SL_RanForGen as rfg_cl
rfg = rfg_cl
print("Supervised Learning: RANDOM FOREST GENERATION \n Results:")
machine, X, y = rfg.read_fit(mainset)
prediction = rfg_cl.rfg_spv_predict(machine, X, y)
print"SL: RFG - Conf. Matrix".center(45,'_'), "\n", prediction, "\n"
pltr.bars(prediction, plt_name="Random Forest Classifier")

import USL_RanForGen as rfg_c
rfg = rfg_c
print("Un-supervised Learning: RANDOM FOREST GENERATION \n Results:")
machine, X, y = rfg.read_fit(mainset)
del y  # No training
pred_df_rf = rfg_c.rfg_usp_predict(machine, X)
print"USL: NBC - Predictions".center(45,'_'), "\n", pred_df_rf.head(), "\n"
pr = pred_df_rf.iloc[:, -1].value_counts()
pltr.biplt(gt, pr, "UnSupervised Random Forest Gen.")

try:
    rfg_cl.plot(machine, X, y)  # UNSTABLE PLOTTING
except Exception as e:
    print "[rfg_c.plot] Plot Warning:", e, "(to be resolved)"