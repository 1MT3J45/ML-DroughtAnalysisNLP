import pandas as pd
import numpy as np
import pickle
import time
import freqWordSelection as fws

# Importing all Classified Data

pdataset = pd.read_csv('TestOnly/positive_tweets.csv', names=['tweet', 'classified'])
ndataset = pd.read_csv('TestOnly/negative_tweets.csv', names=['tweet', 'classified'])
nudataset = pd.read_csv('TestOnly/neutral_tweets.csv', names=['tweet', 'classified'])

# ---------------------------------------

import linecache
import sys

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)


# -------------------------------------------------------------------------------------------- POLARITY JUDGEMENT
from string import punctuation

def tweetPolarity():
    pcount = 0
    ncount = 0
    nucount = 0

    # tweets = open("auto_domain_tweets_drought.csv").read()
    tweets = open("TestOnly/auto_domain_tweets.csv").read()
    tweets_list = tweets.split('\n')

    pos_sent = open("TestOnly/positive.txt").read()
    positive_words = pos_sent.split('\n')
    positive_counts = []

    neg_sent = open('TestOnly/negative.txt').read()
    negative_words = neg_sent.split('\n')
    negative_counts = []

    outFile = open('SDT_pol_sen_result.csv', 'w')
    posFile = open('TestOnly/positive_tweets.csv', 'w')
    negFile = open('TestOnly/negative_tweets.csv', 'w')
    neuFile = open('TestOnly/neutral_tweets.csv', 'w')

    for tweet in tweets_list:
        positive_counter = 0
        negative_counter = 0

        tweet_processed = tweet.lower()

        for p in list(punctuation):
            tweet_processed = tweet_processed.replace(p, '')

        words = tweet_processed.split(' ')
        word_count = len(words)
        for word in words:
            if word in positive_words:
                positive_counter = positive_counter + 1
            elif word in negative_words:
                negative_counter = negative_counter + 1

        positive_counts.append(positive_counter / word_count)
        negative_counts.append(negative_counter / word_count)

        positive_score = positive_counter / word_count
        negative_score = negative_counter / word_count

        classification = ""

        if positive_counter > negative_counter:
            classification = "Positive"
            result = tweet + "," + classification + "\n"
            posFile.write(result)
            pcount = pcount + 1
        elif negative_counter > positive_counter:
            classification = "Negative"
            result = tweet + "," + classification + "\n"
            negFile.write(result)
            ncount = ncount + 1
        else:
            classification = "Neutral"
            result = tweet + "," + classification + "\n"
            neuFile.write(result)
            nucount = nucount + 1

        if tweet == "":
            print("")
        else:

            result = tweet + "," + str(positive_score) + "," + str(negative_score) + "," + classification + "\n"
            outFile.write(result)

        # outFile.close()
        # posFile.close()
        # negFile.close()
        # neuFile.close()

        labelfont = ('times', 10)
        # open file
        ##            label1 = tk.Label(window,width = 40, height = 2,text = "Tweets, relief" , relief = tkinter.RIDGE)
        ##
        ##            label1.place(x = 0, y = 0 )

        ##            label(window, text="Sentiment Polarity").grid(row=0)
        ##            label(window, text="Positive Tweets").grid(row=1)
        ##            label(window, text=pcount).grid(row=1,column=1)
        ##
        ##            label(window, text="Negative Tweets").grid(row=2)
        ##            label(window, text=ncount).grid(row=2,column=1)
        ##
        ##            label(window, text="Neutral Tweets").grid(row=3)
        ##            label(window, text=nucount).grid(row=3,column=1)

    tweets = open("TestOnly/auto_domain_tweets.csv").read()
    tweets_list = tweets.split('\n')

    pos_sent = open("TestOnly/positive.txt").read()
    positive_words = pos_sent.split('\n')
    positive_counts = []

    neg_sent = open('TestOnly/negative.txt').read()
    negative_words = neg_sent.split('\n')
    negative_counts = []

    outFile = open('SDT_pol_sen_result.csv', 'w')

    for tweet in tweets_list:
        positive_counter = 0
        negative_counter = 0

        tweet_processed = tweet.lower()

        for p in list(punctuation):
            tweet_processed = tweet_processed.replace(p, '')

        words = tweet_processed.split(' ')
        word_count = len(words)
        for word in words:
            if word in positive_words:
                positive_counter = positive_counter + 1
            elif word in negative_words:
                negative_counter = negative_counter + 1

        positive_counts.append(positive_counter / word_count)
        negative_counts.append(negative_counter / word_count)

        positive_score = positive_counter / word_count
        negative_score = negative_counter / word_count

        classification = ""

        if positive_counter > negative_counter:
            classification = "Positive"
        elif negative_counter > positive_counter:
            classification = "Negative"
        else:
            classification = "Neutral"

        if tweet == "":
            print("")
        else:

            result = tweet + "," + str(positive_score) + "," + str(negative_score) + "," + classification + "\n"
            outFile.write(result)

    outFile.close()

# -------------------------------------------------------------------------------------------- POLARITY JUDGEMENT

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
# ----------------------------------------------------- EXECUTION SEQ BEGINS HERE ---------------


tweetPolarity()

pos_df = na_remover(pdataset)
neg_df = na_remover(ndataset)
neu_df = na_remover(nudataset)

# Merging Positives and Negatives for Analysis
mainset = pd.concat([pos_df, neg_df, neu_df], axis=0, ignore_index=True)
mainset['classified'] = mainset['classified'].map({'Positive':1, 'Negative':0, 'Neutral':-1})
mainset = mainset.dropna().reset_index(drop=True)   # RESOLVER #01

# SUPERVISED LEARNING ALGORITHM
import SL_NB_Processor as cnp
import plotter as pltr
try:
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
        print tw, "\n", type(all_tweets)     # ISSUE #01 RESOLVED

    # freqD is Frequency of the Words used in Tweets
    freqD = nltk.FreqDist(nltk.word_tokenize(all_tweets))
    for k, v in freqD.items(): # Finding most Freq. words
        if v > 25:
            print k
    # OPTIONAL F IN CASE OF GARBAGE OCCURRED IN EXISTING

    # D is Synonyms
    D = pd.read_csv('TestOnly/synonyms.csv')
    D = list(D.iloc[0:,0])

    # F is Popular tags
    F = pd.read_csv("TestOnly/ptag_area.csv",)
    F = pd.Series(F.iloc[:,0])
    F = list(F)
    # ser_obj = open('pop_tags', 'r')
    # F = pickle.load(ser_obj)

    # -------------------------------------------------------- COMMIT STEP 1
    nltk.download('vader_lexicon')
    # from nltk.sentiment.vader import SentimentIntensityAnalyzer
    # sia = SentimentIntensityAnalyzer()
    # for twe in nltk.word_tokenize(all_tweets):
    #     scores = sia.polarity_scores(text=twe)
    #     print twe
    #     print "POS:", scores.get('pos')
    #     print "NEG:", scores.get('neg')
    #     print "NEU:", scores.get('neu')
    # -------------------------------------------------------- COMMIT STEP 2 (For every word in one tweet)
    fuzzy_df = pd.DataFrame(columns=['tweets', 'classified'])
    for i in range(len(tweets)):
        sent = nltk.word_tokenize(tweets[i])
        PoS_TAGS = nltk.pos_tag(sent)

        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()

        one_sentence = tweets.iloc[i]
        scores = sia.polarity_scores(text=one_sentence)
        # print "POS:", scores.get('pos')
        # print "NEG:", scores.get('neg')
        # print "NEU:", scores.get('neu')

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
                    if tri_pairs[0] or tri_pairs[1] or tri_pairs[2] in F:
                        print("[True]: Tri Pairs are found in Frequent Wordset")
                        if RES is "Positive":
                            RES = "Highly Positive"
                        elif RES is "Negative":
                            RES = "Highly Negative"
                    else:
                        print"[False]: Doesn't Match with Frequent Wordset\n"
                else:
                    print"[False]: Tri Pairs Matched Nowhere in D\n"
            else:
                print "[TriPair(F)]: Pattern for Adverb, Adverb, Adjective did not match.\n " \
                      "Looking for Bi-Pair Patterns\n"
        print(tri_pairs)

        # -------------------------------------------------------- PATTERN ADVERB, ADJECTIVE (Down)
        bi_pairs = list()
        for (w1, tag1), (w2, tag2) in nltk.bigrams(PoS_TAGS):
            if tag1.startswith("RB") and tag2.startswith("JJ"):
                bi_pairs.append((w1, w2))
                if bi_pairs[0] or bi_pairs[1] in D:
                    print("[True]: Bi Pairs are found in Drought Rel. Term")
                    if bi_pairs[0] or bi_pairs[1] in F:
                        print("[True]: Bi Pairs are found in Frequent Wordset")
                        if RES is "Positive":
                            RES = "Moderately Positive"
                        elif RES is "Negative":
                            RES = "Moderately Negative"
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
                    if w in F:
                        print("Matched with F")
                        if RES is "Positive":
                            RES = "Positive"
                        elif RES is "Negative":
                            RES = "Negative"
                    else:
                        print("Couldn't Match with F")
                else:
                    print("the")
            else:
                print w, "is not an ADJECTIVE"

        # -------------------------------------------------------- MAKING ENTRY OF RECORDS OF TWEETS and POLARITY RESULT
        fuzzy_df = fuzzy_df.append({'tweets': tweets[i], 'classified': RES}, ignore_index=True)       # ADDING RECORDS IN DATAFRAME

    fuzzy_df.to_csv("TestOnly/ReFuzzy.csv", index=False)
    fws_df = fws.findFreqWord(fuzzyDF=fuzzy_df)
    sum_df = pd.get_dummies(fws_df[['Classified', 'FreqWord']], columns=['FreqWord']).set_index('Classified').sum(
        level=0)
    sum_df.columns = sum_df.columns.str.split('_').str[1]
    sum_df.to_csv('TestOnly/ClassFreq.csv')

    PS = (fuzzy_df['classified'] == 'Positive').sum()
    H_PS = (fuzzy_df['classified'] == 'Highly Positive').sum()
    M_PS = (fuzzy_df['classified'] == 'Moderately Positive').sum()
    NG = (fuzzy_df['classified'] == 'Negative').sum()
    H_NG = (fuzzy_df['classified'] == 'Highly Negative').sum()
    M_NG = (fuzzy_df['classified'] == 'Moderately Negative').sum()
    text = "Fuzzy Logic Stats"
    pltr.stackplotter(H_NG, M_NG, NG, H_PS, M_PS, PS, text)
    pltr.simple_plot(dataframe=sum_df)

except Exception as e:
    print "[Refuzz]:", e
    PrintException()
