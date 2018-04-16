import pandas as pd
import numpy as np
import pickle
import subprocess
import freqWordSelection as fws

# Importing all Classified Data
pdataset = pd.read_csv('Drought/positive_tweets.csv', names=['tweet', 'classified'])
ndataset = pd.read_csv('Drought/negative_tweets.csv', names=['tweet', 'classified'])
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
mainset['classified'] = mainset['classified'].map({'Positive': 1, 'Negative': 0, 'Neutral': -1})
mainset = mainset.dropna().reset_index(drop=True)  # RESOLVER #01

# SUPERVISED LEARNING ALGORITHM
import SL_NB_Processor as cnp
import plotter as pltr

machine, X, y = cnp.processor(mainset)
print("\nSupervised Learning: Naive Bayes \nResults:")
prediction = cnp.sl_prediction(machine, X, y)
print"SL: NBC - Conf. Matrix".center(45, '_'), "\n", prediction, "\n"
pltr.bars(prediction, plt_name="SL - Naive Bayes Classifier")

# UN-SUPERVISED LEARNING ALGORITHM
import USL_NB_Processor as cnp

machine, X, y = cnp.processor(mainset)
# No Train Test Split and hence no need of y
del y
pred_df = cnp.usl_prediction(machine, X)
print("Un-Supervised Learning: Naive Bayes \n Results:")
print"USL: NBC - Predictions".center(45, '_'), "\n", prediction, "\n"
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
print"SL: RFG - Conf. Matrix".center(45, '_'), "\n", prediction, "\n"
pltr.bars(prediction, plt_name="Random Forest Classifier")

import USL_RanForGen as rfg_c

rfg = rfg_c
print("Un-supervised Learning: RANDOM FOREST GENERATION \n Results:")
machine, X, y = rfg.read_fit(mainset)
del y  # No training
pred_df_rf = rfg_c.rfg_usp_predict(machine, X)
print"USL: NBC - Predictions".center(45, '_'), "\n", pred_df_rf.head(), "\n"
pr = pred_df_rf.iloc[:, -1].value_counts()
pltr.biplt(gt, pr, "UnSupervised Random Forest Gen.")

# ----------------------------- FUZZY KITCHEN -----------------------------
import nltk

nltk.download('averaged_perceptron_tagger')
all_tweets = " "
tweets = mainset.iloc[0:, 0]
print(tweets[0])

for i in range(mainset.__len__()):
    tw = mainset.iloc[i, 0]
    if not isinstance(tw, float):
        all_tweets += tw
    print tw, "\n", type(all_tweets)  # ISSUE #01 RESOLVED

# freqD is Frequency of the Words used in Tweets
freqD = nltk.FreqDist(nltk.word_tokenize(all_tweets))
# for k, v in freqD.items(): # Finding most Freq. words
#     if v > 25:
#         print k
# OPTIONAL F IN CASE OF GARBAGE OCCURRED IN EXISTING

# D is Synonyms
D = pd.read_csv('synonyms.csv')
D = list(D.iloc[0:, 0])

# F is Popular tags
# ser_obj = open('pop_tags', 'r')
# F = pickle.load(ser_obj)
F = pd.read_csv('ptag.csv')
F = list(F.iloc[:, 0])
# -------------------------------------------------------- COMMIT STEP 1
nltk.download('vader_lexicon')

# -------------------------------------------------------- COMMIT STEP 2 (For every word in one tweet)
fuzzy_df = pd.DataFrame(columns=['Tweets', 'Classified'])

######################## TEST CASE 1 #############################
FW = ''
for i in range(len(tweets)):
    sent = nltk.word_tokenize(tweets[i])
    PoS_TAGS = nltk.pos_tag(sent)

    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()

    one_sentence = tweets.iloc[i]
    scores = sia.polarity_scores(text=one_sentence)
    print "POS:", scores.get('pos')
    print "NEG:", scores.get('neg')
    print "NEU:", scores.get('neu')

    POS = scores.get('pos')
    NEG = scores.get('neg')
    NEU = scores.get('neu')
    RES = str()

    if POS > NEG:
        RES = 'Positive'
    elif NEG > POS:
        RES = 'Negative'
    elif NEU >= 0.5 or POS > NEU:
        RES = 'Positive'
    elif NEU < 0.5:
        RES = 'Negative'

    # -------------------------------------------------------- PATTERN ADVERB, ADVERB, ADJECTIVE (Down)
    tri_pairs = list()
    for (w1, tag1), (w2, tag2), (w3, tag3) in nltk.trigrams(PoS_TAGS):
        if tag1.startswith("RB") and tag2.startswith("RB") and tag3.startswith("JJ"):
            tri_pairs.append((w1, w2, w3))
            if tri_pairs[0] or tri_pairs[1] or tri_pairs[2] in D:
                print("[True]: Tri Pairs are found in Drought Rel. Term")
                # TRIGGER AREA
                for j in range(len(F)):
                    if tri_pairs[0] or tri_pairs[1] or tri_pairs[2] in F[j]:
                        print("[True]: Tri Pairs are found in Frequent Wordset")
                        if RES is "Positive":
                            RES = "Highly Positive"
                            FW = F[j]
                            # fuzzy_df['FreqWord'].map(lambda x: next((y for y in x.split() if y in F), 'Not Found'))
                        elif RES is "Negative":
                            RES = "Highly Negative"
                            FW = F[j]
                    else:
                        print"[False]: Doesn't Match with Frequent Wordset\n"

            else:
                print"[False]: Tri Pairs Matched Nowhere in D\n"

        else:
            print "[TriPair(F)]: Pattern for Adverb, Adverb, Adjective did not match.\n Looking for Bi-Pair Patterns\n"
    print(tri_pairs)

    # -------------------------------------------------------- PATTERN ADVERB, ADJECTIVE (Down)
    bi_pairs = list()
    for (w1, tag1), (w2, tag2) in nltk.bigrams(PoS_TAGS):
        if tag1.startswith("RB") and tag2.startswith("JJ"):
            bi_pairs.append((w1, w2))

            if bi_pairs[0] or bi_pairs[1] in D:
                print("[True]: Bi Pairs are found in Drought Rel. Term")

                for k in range(len(F)):
                    if bi_pairs[0] or bi_pairs[1] is F[k]:
                        print("[True]: Bi Pairs are found in Frequent Wordset")
                        if RES is "Positive":
                            RES = "Moderately Positive"
                            FW = F[k]
                        elif RES is "Negative":
                            RES = "Moderately Negative"
                            FW = F[k]
                    else:
                        print("[False]: Bi Pairs found missing in Freq. Wordset")

            else:
                print("[False]: Bi Pairs Matched Nowhere in D")

        else:
            print("[BiPair(F)]: Pattern Not Matched, Looking for Mono Pattern")
    print(bi_pairs)

    # -------------------------------------------------------- PATTERN ADJECTIVE (Down)
    for w, tag in PoS_TAGS:
        print w, " - ", tag
        if tag.startswith("JJ"):
            if w in D:
                print("Matched with D")
                for l in range(len(F)):
                    if w is F[l]:
                        print("Matched with F")
                        if RES is "Positive":
                            RES = "Positive"
                            FW = F[l]
                        elif RES is "Negative":
                            RES = "Negative"
                            FW = F[l]
                    else:
                        print("Unmatched in F")
                        FW = F[l] in sent
            else:
                print("Unmatched in D")
        else:
            print w, "is not an ADJECTIVE"

    # -------------------------------------------------------- MAKING ENTRY OF RECORDS OF TWEETS and POLARITY RESULT
    fuzzy_df = fuzzy_df.append({'Tweets': tweets[i], 'Classified': RES}, ignore_index=True)
# ADDING RECORDS IN DATAFRAME
fuzzy_df.to_csv("fuzzy.csv", index=False)

import freqWordSelection as fws

fws_df = fws.findFreqWord(fuzzyDF=fuzzy_df)
sum_df = pd.get_dummies(fws_df[['Classified', 'FreqWord']], columns=['FreqWord']).set_index('Classified').sum(level=0)
sum_df.columns = sum_df.columns.str.split('_').str[1]
# sum_df = pd.crosstab(fws_df.Classified, fws_df.FreqWord)

PS = (fuzzy_df['Classified'] == 'Positive').sum()
H_PS = (fuzzy_df['Classified'] == 'Highly Positive').sum()
M_PS = (fuzzy_df['Classified'] == 'Moderately Positive').sum()
NG = (fuzzy_df['Classified'] == 'Negative').sum()
H_NG = (fuzzy_df['Classified'] == 'Highly Negative').sum()
M_NG = (fuzzy_df['Classified'] == 'Moderately Negative').sum()
text = "Fuzzy Logic Stats"
pltr.stackplotter(H_NG, M_NG, NG, H_PS, M_PS, PS, text)
pltr.simple_plot(dataframe=sum_df)
