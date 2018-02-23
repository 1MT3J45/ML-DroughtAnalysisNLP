from __future__ import division
from __future__ import print_function
from __future__ import with_statement
from string import punctuation
from tkintertable.Tables import TableCanvas
from tkintertable.TableModels import TableModel
from Tkinter import *
from replacers import SpellingReplacer
from itertools import izip
from tweepy import Stream
from tweepy.streaming import StreamListener
from wordsegment import load, segment
import io
import tweepy
import shutil
import nltk
import pandas as pd
import re
import csv
import uuid
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from wordsegment import segment
from nltk.corpus import PlaintextCorpusReader
from replacers import *
from nltk.metrics import edit_distance
from replacers import CsvWordReplacer
from nltk.corpus import stopwords
import random, sys
from nltk.corpus import names
from textblob import TextBlob, Word, Blobber
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import os
import json
import time, datetime
import ConfigParser
# import oauth2 as oauth
import urllib2 as urllib
from time import sleep
from threading import Thread
from nltk import*
from nltk.corpus import*
import string
import itertools
from prettytable import PrettyTable
import matplotlib
from matplotlib import *
import numpy as np
import matplotlib.pyplot as plt
import urllib
import collections
import ttk
import tkMessageBox
from ttk import Frame, Style

# Tkinter
try:
    import Tkinter as tk     ## Python 2.x
except ImportError:
    import tkinter as tk     ## Python 3.x 


class TweetimentFrame(tk.Frame):
##    """
##        This is the base class for the program.
##    """
   
    # dictionary for storing configuration data
    config = {}
    

    

    def __init__(self, parent):
##        """
##            Initialize the parent Frame
##        """
        
        tk.Frame.__init__(self, parent)            
        self.parent = parent
        self.initUI()
        


    def initUI(self):
##        """
##            Method for initializing the root window elements. 
##            All root level buttons and labels are initialized here.
##        """
        
        self.parent.title("Domain Sentiment Analyzer")
        self.pack(fill=tk.BOTH, expand=1)
##        frame1 = Frame(self.parent, bg="MediumPurple1")
##        frame1.pack()
        TweetimentCanvas = tk.Canvas(self.parent, height=130, width=1000,bg="MediumPurple1")
        TweetimentCanvas.create_text(500, 50, font=("Georgia", 20), text = "Domain Specific Sentiment Analysis")
        TweetimentCanvas.create_text(500, 100, font=("Georgia", 10), text = "Sentiment Analyzer")
        TweetimentCanvas.place(x = 150, y = 40, width = 1000, height = 130)

        
        global tgui,pcount,ncount,nucount
        
        Search=tk.Label(self.parent, text="Enter Search Domain")
        Search.place(x = 50, y = 200, width = 200, height = 30)

        self.s1 = StringVar(self.parent)
        domain = tk.Entry(self.parent, textvariable=self.s1)
        domain.place(x = 210, y = 200, width = 200, height = 30)
        

        self.flag=StringVar(self.parent)
        self.flag="none"
        

        # button for Downloadtweets
        RunDownloadButton = tk.Button(self.parent, text = "Download Tweets", command =self.downloadTweet, bg="blue", fg="white")
        RunDownloadButton.place(x = 430, y = 200, width = 200, height = 30)

        # button for Tweets Repository
        RunRepositoryButton = tk.Button(self.parent, text = "Tweets Repository", command =self.tweetRepository, bg="blue", fg="white")
        RunRepositoryButton.place(x = 650, y = 200, width = 200, height = 30)
       

        # button for preprocess tweets
        RunTweetPreprocessButton = tk.Button(self.parent, text = "Preprocess Tweets", command = self.preprocessTweet, bg="blue", fg="white")
        RunTweetPreprocessButton.place(x = 50, y = 320, width = 200, height = 30)

        # button for extracting tweets 
        RunExtractButton = tk.Button(self.parent, text = "Extract Tweets", command = self.extractTweet, bg="blue", fg="white")
        RunExtractButton.place(x = 300, y = 320, width = 200, height = 30)

        # button for tweet polarity
        RunPolarityButton = tk.Button(self.parent, text = "Tweet Polarity", command = self.tweetPolarity, bg="blue", fg="white")
        RunPolarityButton.place(x = 550, y = 320, width = 200, height = 30)

        # button for tweet Ngram
        RunNgramButton = tk.Button(self.parent, text = "N-Gram Analysis", command = self.Ngram, bg="blue", fg="white")
        RunNgramButton.place(x = 800, y = 320, width = 200, height = 30)

        # button for tweet polarity
        RunAssoButton = tk.Button(self.parent, text = "Paired Association", command = self.wordAsso, bg="blue", fg="white")
        RunAssoButton.place(x = 1050, y = 320, width = 200, height = 30)

        # button for Popular tags
        RunPtagButton = tk.Button(self.parent, text = "Popular Tags", command = self.Ptag, bg="brown", fg="white")
        RunPtagButton.place(x = 50, y = 380, width = 200, height = 30)

        # button for Tag tweets
        RunTagButton = tk.Button(self.parent, text = "Tag Tweets", command = self.tagTweets, bg="brown", fg="white")
        RunTagButton.place(x = 300, y = 380, width = 200, height = 30)

        # button for quit
        RunQuitButton = tk.Button(self.parent, text = "Quit", command = root.destroy, bg="red", fg="black")
        RunQuitButton.place(x = 50, y = 450, width = 200, height = 30)

        




###################################################################################################################
    def downloadTweet (self) :
        self.flag="download"
        print(self.flag)
        auth = tweepy.auth.OAuthHandler('HOtua649r1m7Xopbc4rIdMWNh', '6YcJMtmbrQCXS0ur0SYJ00y4LCR567PxqZo0MK3uPrBavTPaao')
        auth.set_access_token('3018145817-rwPn950cioS2Nau0MLwtVTlnQhefIHLID1AXo6f', 'WuLRRYrQbpvGd7jmxGWJZY2NzBh28nJyXlq6nrC3H1V9B')

        api = tweepy.API(auth)

        # Open/create a file to append data to
        csvFile = open('download.csv', 'w')

        #Use csv writer
        csvWriter = csv.writer(csvFile)

       
        for tweet in tweepy.Cursor(api.search,q = self.s1.get(),lang = "en",n=5000).items():

            # Write a row to the CSV file. I use encode UTF-8
            csvWriter.writerow([tweet.text.encode('utf-8')])
            print (tweet.created_at, tweet.text)
           

        csvFile.close()

##################################################################################################################

    def tweetRepository (self) :
        #repoFile = open('repository.csv','r')
        self.flag="repo"
        print(self.flag)

###################################################################################################################

    def preprocessTweet (self) :
        #self.flag="download"
        if self.flag=="download":
            
        #Module 1:Eliminate duplicate tweets from the file auto_tweet.csv and copy unique tweets in auto_p1.csv

            inFile = open('download.csv','r')
            #inFile = open('FinalCorpus.csv','r')

            outFile = open('auto_p1.csv','w')

            listLines = []

            for line in inFile:

                if line in listLines:
                    continue

                else:
                    outFile.write(line)
                    listLines.append(line)

            outFile.close()

            inFile.close()


        #Module 2:Slang word replacement


            s = open("auto_p1.csv", "r")
            s1=open("auto_p2.csv","w")
            list1=[]
            word=""
#wr = csv.writer(s1, dialect='excel')
            listLines1 = []
            new_list1=[]
            new_list2=[" "]

            replacerSlang = CsvWordReplacer('SDT_slang_and_acronym.csv')

            for line in s:
                replacer = SpellingReplacer()
                replacerSlang = CsvWordReplacer('SDT_slang_and_acronym.csv')
                listLines1=segment(line)
                list1=[]
                for word_list in listLines1:
                    word_list=replacerSlang.replace(word_list)
        #print word_list
                    list1=list1+list(replacer.replace(word_list))+new_list2
                    my_lst_str = ''.join(list1)
                s1.write(my_lst_str)
                s1.write("\n")
    #wr.writerow(my_lst_str)
            
            s.close()
            s1.close()


#Spell correction

            s = open("auto_p2.csv", "r")
            s1=open("auto_p3.csv","w")
            list1=[]
            word=""
#wr = csv.writer(s1, dialect='excel')
            listLines1 = []
            new_list1=[]
            new_list2=[" "]
    #us_replacer = SpellingReplacer('en_US')
    #replacerSlang = CsvWordReplacer('SDT_slang_and_acronym.csv')

            for line in s:
                us_replacer = SpellingReplacer('en_US')
        #replacerSlang = CsvWordReplacer('SDT_slang_and_acronym.csv')
                listLines1=segment(line)
                list1=[]
                for word_list in listLines1:
                    word_list=us_replacer.replace(word_list)
        #print word_list
                    list1=list1+list(replacer.replace(word_list))+new_list2
                    my_lst_str = ''.join(list1)
                s1.write(my_lst_str)
                s1.write("\n")
    #wr.writerow(my_lst_str)
            
            s.close()
            s1.close()


#Read the tweets one by one and process it

            fp = open('auto_p3.csv', 'r')
            fpw=open('auto_clean.csv','w')
            tweetLine = fp.readline()

            while tweetLine:
                # process the tweets

                #Convert to lower case
                tweetLine = tweetLine.lower()

                #Convert @username to AT_USER
                tweetLine = re.sub('rt','',tweetLine)

                #Convert @username to AT_USER
                tweetLine = re.sub('@[^\s]+','',tweetLine)
    
                #Convert www.* or https?://* to URL
                tweetLine = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',"",tweetLine)
    
                #Remove at people
                tweetLine = re.sub("@\\w+", "",tweetLine)
   
                #Remove additional white spaces
                tweetLine = re.sub('[\s]+', ' ', tweetLine)
    
                #Replace #word with word
                tweetLine = re.sub(r'#([^\s]+)', r'\1', tweetLine)

                #Remove retweet entities
                tweetLine=re.sub("(RT|via)((?:\\b\\W*@\\w+)+)", "",tweetLine)

                # remove punctuation
                tweetLine=re.sub("[[:punct:]]", "",tweetLine)

            # remove numbers
            #tweet=re.sub("[[:digit:]]", "",tweet)

                # remove html links
                tweetLine=re.sub("http\\w+", "",tweetLine)

                # remove unnecessary spaces
                tweetLine=re.sub("[ \t]{2,}", "",tweetLine)
                tweetLine=re.sub("^\\s+|\\s+$", "",tweetLine)

                #trim
                tweetLine = tweetLine.strip('\'"')

                #try
                tweetLine=re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweetLine)

                #Remove quotes
                tweetLine = re.sub(r'&amp;quot;|&amp;amp','',tweetLine)
                  
                #Remove citations
                tweetLine = re.sub(r'@[a-zA-Z0-9]*', '',tweetLine)
                  
                #Remove tickers
                tweetLine = re.sub(r'\$[a-zA-Z0-9]*', '',tweetLine)
                  
                #Remove numbers
                tweetLine = re.sub(r'[0-9]*','',tweetLine)

                #Remove numbers
                tweetLine = re.sub(r'amp','',tweetLine)
                processedTweet = tweetLine
                fpw.writelines(processedTweet)
                fpw.write("  ")
                fpw.write("\n")
    
                tweetLine = fp.readline()
#end loop
           
        else:
                #Module 1:Eliminate duplicate tweets from the file auto_tweet.csv and copy unique tweets in auto_p1.csv

            inFile = open('FinalCorpus.csv','r')

            outFile = open('auto_p1.csv','w')

            listLines = []

            for line in inFile:

                if line in listLines:
                    continue

                else:
                    outFile.write(line)
                    listLines.append(line)

            outFile.close()

            inFile.close()


        #Module 2:Slang word replacement


            s = open("auto_p1.csv", "r")
            s1=open("auto_p2.csv","w")
            list1=[]
            word=""
#wr = csv.writer(s1, dialect='excel')
            listLines1 = []
            new_list1=[]
            new_list2=[" "]

            replacerSlang = CsvWordReplacer('SDT_slang_and_acronym.csv')

            for line in s:
                replacer = SpellingReplacer()
                replacerSlang = CsvWordReplacer('SDT_slang_and_acronym.csv')
                listLines1=segment(line)
                list1=[]
                for word_list in listLines1:
                    word_list=replacerSlang.replace(word_list)
        #print word_list
                    list1=list1+list(replacer.replace(word_list))+new_list2
                    my_lst_str = ''.join(list1)
                s1.write(my_lst_str)
                s1.write("\n")
    #wr.writerow(my_lst_str)
            
            s.close()
            s1.close()


#Spell correction

            s = open("auto_p2.csv", "r")
            s1=open("auto_p3.csv","w")
            list1=[]
            word=""
#wr = csv.writer(s1, dialect='excel')
            listLines1 = []
            new_list1=[]
            new_list2=[" "]
    #us_replacer = SpellingReplacer('en_US')
    #replacerSlang = CsvWordReplacer('SDT_slang_and_acronym.csv')

            for line in s:
                us_replacer = SpellingReplacer('en_US')
        #replacerSlang = CsvWordReplacer('SDT_slang_and_acronym.csv')
                listLines1=segment(line)
                list1=[]
                for word_list in listLines1:
                    word_list=us_replacer.replace(word_list)
        #print word_list
                    list1=list1+list(replacer.replace(word_list))+new_list2
                    my_lst_str = ''.join(list1)
                s1.write(my_lst_str)
                s1.write("\n")
    #wr.writerow(my_lst_str)
            
            s.close()
            s1.close()


#Read the tweets one by one and process it

            fp = open('auto_p3.csv', 'r')
            fpw=open('auto_clean.csv','w')
            tweetLine = fp.readline()

            while tweetLine:
                # process the tweets

                #Convert to lower case
                tweetLine = tweetLine.lower()

                #Convert @username to AT_USER
                tweetLine = re.sub('rt','',tweetLine)

                #Convert @username to AT_USER
                tweetLine = re.sub('@[^\s]+','',tweetLine)
    
                #Convert www.* or https?://* to URL
                tweetLine = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',"",tweetLine)
    
                #Remove at people
                tweetLine = re.sub("@\\w+", "",tweetLine)
   
                #Remove additional white spaces
                tweetLine = re.sub('[\s]+', ' ', tweetLine)
    
                #Replace #word with word
                tweetLine = re.sub(r'#([^\s]+)', r'\1', tweetLine)

                #Remove retweet entities
                tweetLine=re.sub("(RT|via)((?:\\b\\W*@\\w+)+)", "",tweetLine)

                # remove punctuation
                tweetLine=re.sub("[[:punct:]]", "",tweetLine)

            # remove numbers
            #tweet=re.sub("[[:digit:]]", "",tweet)

                # remove html links
                tweetLine=re.sub("http\\w+", "",tweetLine)

                # remove unnecessary spaces
                tweetLine=re.sub("[ \t]{2,}", "",tweetLine)
                tweetLine=re.sub("^\\s+|\\s+$", "",tweetLine)

                #trim
                tweetLine = tweetLine.strip('\'"')

                #try
                tweetLine=re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweetLine)

                #Remove quotes
                tweetLine = re.sub(r'&amp;quot;|&amp;amp','',tweetLine)
                  
                #Remove citations
                tweetLine = re.sub(r'@[a-zA-Z0-9]*', '',tweetLine)
                  
                #Remove tickers
                tweetLine = re.sub(r'\$[a-zA-Z0-9]*', '',tweetLine)
                  
                #Remove numbers
                tweetLine = re.sub(r'[0-9]*','',tweetLine)


                #Remove numbers
                tweetLine = re.sub(r'amp','',tweetLine)

                
##
##                
                processedTweet = tweetLine
                fpw.writelines(processedTweet)
                fpw.write("  ")
                fpw.write("\n")
    
                tweetLine = fp.readline()
                
#end loop
            

        fp.close()
        fpw.close()



            # removing stopwords 
        stop_words = set(stopwords.words("english"))
        my_lst=""
        with open('auto_clean.csv','r') as inFile, open('auto_clean_stopword.csv','w') as outFile:
            for line in inFile.readlines():
                    #line1=self.wordSepereter(line)
                words=line.split()
                for word in words:
##                       
                    if not word in stop_words:
                        my_lst = ''.join(word)
                        outFile.write(my_lst)
                        outFile.write(" ")
                outFile.write("\n")
        inFile.close()
        outFile.close()
        fr=open('auto_clean_stopword.csv','r')
            
        fw=open('auto_clean_sep.csv','w')
        for line in fr.readlines():
            result=self.wordSepereter(line)
            fw.write(result)
            fw.write("\n")
        fr.close()
        fw.close()
            
               
            

            
##########################################################################################################
    
    def wordSepereter(self,line):
        from wordsegment import load, segment
        load()
        my_lst=""
        s=segment(line)
        for i in s:
            my_lst = my_lst+" "+''.join(i)
        return (my_lst)
###################################################################################################################

    def extractTweet (self) :

#Module 1:Eliminate duplicate tweets from the file auto_tweet.csv and copy unique tweets in auto_clean_final.csv

        shutil.copy2('auto_clean_sep.csv', 'auto_clean_copy.csv')
    
        with open('auto_clean_sep.csv', 'r') as file1:
            with open('auto_clean_copy.csv', 'r') as file2:
                same = set(file1).intersection(file2)

        same.discard('\n')

        with open('auto_clean_final.csv', 'w') as file_out:
            for line in same:
                file_out.write(line)

        file1.close()
        file2.close()
        file_out.close()

#Pick the tweets based on synonym


        #if (domainVar.get()=='drought')or (domainVar.get()=='Drought'):
        f1 = open('auto_clean_final.csv', 'r')
        f2 = file('synonyms.csv', 'r')
        f3 = file('auto_domain_tweets.csv', 'w')


        c2 = csv.reader(f2)
        c3 = csv.writer(f3)

        masterlist = list(c2)

        count=0

        for hosts_row in f1:
            row = 1
            found = False
            t1=hosts_row.split()
            for master_row in masterlist:
                for word in t1:
            
                    if (word == master_row[0] and self.s1.get()==master_row[1]):
                
                        count=count+1
                        f3.write(hosts_row)
                        print ("yes")
                
                        found = True
                        break
                    row = row + 1
                    #print count
                if not found:
                    print ("no")
                
        print (count)
        f1.close()
        f2.close()
        f3.close()

#################################################################################################################

    def tweetPolarity(self):
        pcount=0
        ncount=0
        nucount=0
        if self.flag=="repo":
            
            window = tk.Toplevel(self)

            #tweets = open("auto_domain_tweets_drought.csv").read()
            tweets = open("auto_domain_tweets.csv").read()
            tweets_list = tweets.split('\n')
        

            pos_sent = open("positive.txt").read()
            positive_words=pos_sent.split('\n')
            positive_counts=[]

            neg_sent = open('negative.txt').read()
            negative_words=neg_sent.split('\n')
            negative_counts=[]

            outFile = open('SDT_pol_sen_result.csv','w')
            posFile = open('positive_tweets.csv','w')
            negFile = open('negative_tweets.csv','w')
            neuFile = open('neutral_tweets.csv','w')

            for tweet in tweets_list:
                positive_counter=0
                negative_counter=0
    
                tweet_processed=tweet.lower()
    
    
                for p in list(punctuation):
                    tweet_processed=tweet_processed.replace(p,'')

                words=tweet_processed.split(' ')
                word_count=len(words)
                for word in words:
                    if word in positive_words:
                        positive_counter=positive_counter+1
                    elif word in negative_words:
                        negative_counter=negative_counter+1
        
                positive_counts.append(positive_counter/word_count)
                negative_counts.append(negative_counter/word_count)

                positive_score=positive_counter/word_count
                negative_score=negative_counter/word_count
    
                classification=""
    
                if positive_counter>negative_counter:
                    classification="Positive"
                    result=tweet+ "," + classification +"\n"
                    posFile.write(result)
                    pcount=pcount+1
                elif negative_counter>positive_counter:
                    classification="Negative"
                    result=tweet + "," + classification +"\n"
                    negFile.write(result)
                    ncount=ncount+1
                else:
                    classification="Neutral"
                    result=tweet + "," + classification +"\n"
                    neuFile.write(result)
                    nucount=nucount+1

    
                if tweet=="":
                    print("")
                else:
                
                    result=tweet + "," + str(positive_score) + "," + str(negative_score) + "," + classification + "\n"
                    outFile.write(result)

            outFile.close()
            posFile.close()
            negFile.close()
            neuFile.close()

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
            
            
        
            label2 = tk.Label(window, width =70 , height = 2,text = "Sentiment ")
            label2 .place(x = 1, y = 1)
            label2 = tk.Label(window, width =18 , height = 2,text = "Sentiment Count")
            label2 .place(x = 30, y = 1)
          
            label3 = tk.Label(window, width =70 , height = 2,text ="Positive Tweets" )
            label3 .place(x = 1, y = 30)
            label4 = tk.Label(window, width =18 , height = 2,text = pcount)
            label4 .place(x = 30, y = 30)
            
            label5 = tk.Label(window, width =70 , height = 2,text = "Negative Tweets")
            label5 .place(x = 1, y = 60)
            label6 = tk.Label(window, width =18 , height = 2,text = ncount)
            label6 .place(x = 30, y = 60)
            
            label7 = tk.Label(window, width =70 , height = 2,text = "Neutral Tweets")
            label7 .place(x = 1, y = 90)
            label8 = tk.Label(window, width =18 , height = 2,text = nucount)
            label8 .place(x = 30, y = 90)
            
##            with open("SDT_pol_sen_result.csv","r") as filee:
##            #file1.seek(0)
##                reader = csv.reader(filee)
##           
##                r = 0
##                for col in reader:
##                    c = 0
##                    w=40
##                
##                    for row in col:
##         
##                        label = tkinter.Label(window, width =w , height = 2,text = row, relief = tkinter.RIDGE)
##                        label.config(bg='black', fg='white')  
##                        label.config(font=labelfont)        
##                        label.grid(row = r, column = c)
##                    #label.pack()
##                        w=20
##                    
##                        c += 1
##                    r += 1
        else:
            window = tk.Toplevel(self)

            tweets = open("auto_domain_tweets.csv").read()
            tweets_list = tweets.split('\n')
        

            pos_sent = open("positive.txt").read()
            positive_words=pos_sent.split('\n')
            positive_counts=[]

            neg_sent = open('negative.txt').read()
            negative_words=neg_sent.split('\n')
            negative_counts=[]

            outFile = open('SDT_pol_sen_result.csv','w')

            for tweet in tweets_list:
                positive_counter=0
                negative_counter=0
    
                tweet_processed=tweet.lower()
    
    
                for p in list(punctuation):
                    tweet_processed=tweet_processed.replace(p,'')

                words=tweet_processed.split(' ')
                word_count=len(words)
                for word in words:
                    if word in positive_words:
                        positive_counter=positive_counter+1
                    elif word in negative_words:
                        negative_counter=negative_counter+1
        
                positive_counts.append(positive_counter/word_count)
                negative_counts.append(negative_counter/word_count)

                positive_score=positive_counter/word_count
                negative_score=negative_counter/word_count
    
                classification=""
    
                if positive_counter>negative_counter:
                    classification="Positive"
                elif negative_counter>positive_counter:
                    classification="Negative"
                else:
                    classification="Neutral"

    
                if tweet=="":
                    print("")
                else:
                
                    result=tweet + "," + str(positive_score) + "," + str(negative_score) + "," + classification + "\n"
                    outFile.write(result)

            outFile.close()

            labelfont = ('times', 10)
        # open file
            label1 = tk.Label(window,width = 40, height = 2,text = "Tweets, relief" , relief = tkinter.RIDGE)
        
            label1.place(x = 0, y = 0 )
        
            label2 = tk.Label(window, width =20 , height = 2,text = "Sentiment ", relief = tkinter.RIDGE)
            label2 .place(x = 41, y = 3)
            with open("SDT_pol_sen_result.csv","r") as filee:
            #file1.seek(0)
                reader = csv.reader(filee)
           
                r = 0
                for col in reader:
                    c = 0
                    w=40
                
                    for row in col:
         
                        label = tkinter.Label(window, width =w , height = 2,text = row, relief = tkinter.RIDGE)
                        label.config(bg='black', fg='white')  
                        label.config(font=labelfont)        
                        label.grid(row = r, column = c)
                    #label.pack()
                        w=20
                    
                        c += 1
                    r += 1
            
          
    
#################################################################################################################

    def Ngram (self) :
        print("")
        print ('--------------------The unigram of tweets-------------------')

        f = open("auto_domain_tweets.csv", "r")
        fw=open("uni_tweets.csv","w")

        tweetlines = f.readlines()

        for line1 in tweetlines:
            
            words = line1.split()
            for word in words:
                unigram_tweet=word
                fw.write(unigram_tweet)
                fw.write("\n")

        f.close()
        fw.close()

        f1 = file('uni_tweets.csv', 'r')
        f2 = file('unigram_corpus.csv', 'r')
        f3 = file('unigramresults.csv', 'w')

        c1 = csv.reader(f1)
        c2 = csv.reader(f2)
        c3 = csv.writer(f3)

        polarity=""
        uneg_score=0
        upos_score=0
        count=0
        flag=1
        one=1
        total_tweet=0

        masterlist = list(c2)
        t = PrettyTable(['Unigram tweets', 'Polarity'])
        for hosts_row in c1:
            total_tweet=total_tweet+1
            row = 1
            found = False
            for master_row in masterlist:
                results_row = hosts_row
                if hosts_row[0] == master_row[0]:
                    count=count+1
                                 
                    if flag == int(master_row[1]):
                        uneg_score=uneg_score+1
                        polarity="Negative"
                    else:
                        upos_score=upos_score+1
                        polarity="Positive"
           
                    t.add_row([hosts_row[0],polarity])
                    uni_result=hosts_row[0] + "," + polarity +"\n"
                    f3.write(uni_result)
                    break
        
                row = row + 1
        
        releventRecordRet=count
        releventRecordNRet=0
        irrRecordRet=total_tweet-releventRecordRet


        Recall = float(releventRecordRet / (releventRecordRet + releventRecordNRet)) * 100
        Precision = float(releventRecordRet / (releventRecordRet + irrRecordRet)) * 100



        print(t)
        print ('Total tweets is ' + str(total_tweet))
        print ('Total relevent tweet is ' + str(count))

        upos_score=float((upos_score)/(count))*float(100)
        print("Positive opinion = " + str(upos_score) + "%")
        uneg_score=(uneg_score/count)*100
        print("Negative opinion = " +str(uneg_score)+ "%")
        print("\n")

        print("Rcall= " + str(Recall))
        print("Precision= " + str(Precision))

        f1.close()
        f2.close()
        f3.close()

#__________________________________________________________________________________________________#

        print ('--------------------The Bigram of tweets-------------------')

        f = open("auto_domain_tweets.csv", "r")
        fpw=open("bi_tweet.csv","w")


    #token=text

        tweetlines = f.readlines()

        for token in tweetlines:
            st=token
            text=nltk.word_tokenize(st)
            token=text
            n = 0
            while n < len(token) - 1:

                fpw.write(token[n]+" " +token[n+1])
                fpw.write("\n")
                n += 2


        fpw.close()
        f.close()



        f1 = file('bi_tweet.csv', 'r')
        f2 = file('bigram_corpus.csv', 'r')
        f3 = file('bigramresults.csv', 'w')

        c1 = csv.reader(f1)
        c2 = csv.reader(f2)
        c3 = csv.writer(f3)

        polarity=""
        bneg_score=0
        bpos_score=0
        count=0
        flag=1
        one=1
        total_tweet=0
        hrow=[]
        mrow=[]




        masterlist = list(c2)
        t = PrettyTable(['Bigram tweets', 'Polarity'])
        for hosts_row in c1:
            total_tweet=total_tweet+1
            row = 1
            found = False
            for master_row in masterlist:
                results_row = hosts_row
                hrow=hosts_row[0]
                mrow=master_row[0]

                if hosts_row[0] == master_row[0]:
                    count=count+1
            
                       
                    if flag == int(master_row[1]):
                        bneg_score=bneg_score+1
                        polarity="Negative"
                    else:
                        bpos_score=bpos_score+1
                        polarity="Positive"
           
                    t.add_row([hosts_row[0],polarity])
                    bi_result=hosts_row[0] + "," + polarity +"\n"
                    f3.write(bi_result)
                    break
        
                row = row + 1

        releventRecordRet=count
        releventRecordNRet=0
        irrRecordRet=total_tweet-releventRecordRet


        print("releventRecordRet=",releventRecordRet,"releventRecordNRet=",releventRecordNRet,"irrRecordRet=",irrRecordRet)


        try:
            Recall = float(releventRecordRet / (releventRecordRet + releventRecordNRet)) * 100
            Precision = float(releventRecordRet / (releventRecordRet + irrRecordRet)) * 100
        except ZeroDivisionError:
            print("division by zero!")
        else:


            print(t)
            print ('Total tweets is ' + str(total_tweet))
            print ('Total relevent tweet is ' + str(count))

            bpos_score=float((bpos_score)/(count))*float(100)
            print("Positive opinion = " + str(bpos_score) + "%")
            bneg_score=(bneg_score/count)*100
            print("Negative opinion = " +str(bneg_score)+ "%")
            print("\n")

            print("Rcall= " + str(Recall))
            print("Precision= " + str(Precision))

        finally:
            print("Bigram Done")

        f1.close()
        f2.close()
        f3.close()

#______________________________________________________________________________________



        print ('--------------------The Trigram of tweets-------------------')

        f = open("auto_domain_tweets.csv", "r")
        fpw=open("tri_tweet.csv","w")


        token=text

        tweetlines = f.readlines()

        for token in tweetlines:
            st=token
            text=nltk.word_tokenize(st)
            token=text
            n = 0
            while n < len(token) - 2:

                fpw.write(token[n]+" " +token[n+1]+" " +token[n+2])
                fpw.write("\n")
                n += 2


        fpw.close()
        f.close()



        f1 = file('tri_tweet.csv', 'r')
        f2 = file('trigram_corpus.csv', 'r')
        f3 = file('trigramresults.csv', 'w')

        c1 = csv.reader(f1)
        c2 = csv.reader(f2)
        c3 = csv.writer(f3)

        polarity=""
        tneg_score=0
        tpos_score=0
        count=0
        flag=1
        one=1
        total_tweet=0
        hrow=[]
        mrow=[]




        masterlist = list(c2)
        t = PrettyTable(['Trigram tweets', 'Polarity'])
        for hosts_row in c1:
            total_tweet=total_tweet+1
            row = 1
            found = False
            for master_row in masterlist:
                results_row = hosts_row
                hrow=hosts_row[0]
                mrow=master_row[0]
                if hosts_row[0] == master_row[0]:
                    count=count+1
            
                       
                    if flag == int(master_row[1]):
                        tneg_score=tneg_score+1
                        polarity="Negative"
                    else:
                        tpos_score=tpos_score+1
                        polarity="Positive"
           
                    t.add_row([hosts_row[0],polarity])
                    tri_result=hosts_row[0] + "," + polarity +"\n"
                    f3.write(tri_result)
                    break
        
                row = row + 1
    

        releventRecordRet=count
        releventRecordNRet=0
        irrRecordRet=total_tweet-releventRecordRet


        print("releventRecordRet=",releventRecordRet,"releventRecordNRet=",releventRecordNRet,"irrRecordRet=",irrRecordRet)

        try:
            Recall = float(releventRecordRet / (releventRecordRet + releventRecordNRet)) * 100
            Precision = float(releventRecordRet / (releventRecordRet + irrRecordRet)) * 100

        except ZeroDivisionError:
            print("division by zero!")

        else:
            print(t)
            print ('Total tweets is ' + str(total_tweet))
            print ('Total relevent tweet is ' + str(count))

            tpos_score=float((tpos_score)/(count))*float(100)
            print("Positive opinion = " + str(tpos_score) + "%")
            tneg_score=(tneg_score/count)*100
            print("Negative opinion = " +str(tneg_score)+ "%")
            print("\n")

            print("Rcall= " + str(Recall))
            print("Precision= " + str(Precision))

        finally:
            print("Trigram Done")

        f1.close()
        f2.close()
        f3.close()



#___________________________________________________________________________________________________________

# data to plot
        n_groups = 3
        means_pos = (upos_score,bpos_score,tpos_score)
        means_neg = (uneg_score,bneg_score,tneg_score)
 
# create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8
 
        rects1 = plt.bar(index, means_pos, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Positive')
 
        rects2 = plt.bar(index + bar_width, means_neg, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Negative')
 
        plt.xlabel('Gram Model')
        plt.ylabel('Polarity Scores')
        plt.title('Polarity analysis based on User Specific Domain')
        plt.xticks(index + bar_width, ('Unigram', 'Bigram', 'Trigram'))
        plt.legend()

        for rect in rects1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,'%d' % int(height),ha='center', va='bottom')

        #autolabel(rects1)
        #autolabel(rects2)

        for rect in rects2:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,'%d' % int(height),ha='center', va='bottom')


        plt.tight_layout()
        plt.show()

#######################################################################################################################

    def wordAsso (self) :
        window = tk.Toplevel(self)
        fpt=open("auto_domain_tweets.csv","r")
        fp=open("P_association.csv","w")
        fn=open("N_association.csv","w")

        tweetlines = fpt.readlines()
        for token in tweetlines:
            s = token
            i=0
            target = self.s1.get()
            words = s.split()
            sentence_length=len(words)
            for i,w in enumerate(words):
            #for i in sentence_length
            
##                if w == target:
##                    ndiff=sentence_length-i
##                    pdiff=i-sentence_length
##                    if ndiff>=3:
##                
##                # next word
##                        fn.write(target + " " + words[i+1] +" " + words[i+2])
##                        fn.write("\n")
##                    elif pdiff<=3:
##                    # previous word
##               
##                        fp.write(words[i-2]+" " + words[i-1]+ " " +target )
##                        fp.write("\n")
##                    else:
##                        print()
                if w == target:
                    ndiff=sentence_length-i
                    pdiff=i-sentence_length
                    if ndiff>=3:
                
                # next word
                        fn.write(target + " " + words[i+1])
                        fn.write("\n")
                    elif pdiff<=3:
                    # previous word
               
                        fp.write(words[i-1]+ " " +target )
                        fp.write("\n")
                    else:
                        print()

        fp.close()
        labelfont = ('times', 10)
        # open file
        label1 = tk.Label(window,width = 40, height = 2,text = "Word Association" , relief = tkinter.RIDGE)
        
        label1.place(x = 0, y = 0 )
        
##        label2 = tk.Label(window, width =20 , height = 2,text = "Word Association", relief = tkinter.RIDGE)
##        label2.place(x = 41, y = 3)
        with open("P_association.csv","r") as filee:
            #file1.seek(0)
            reader = csv.reader(filee)
           
            r = 1
            c = 0
            w=40
                
            for row in reader:
         
                label = tkinter.Label(window, width =w , height = 2,text = row, relief = tkinter.RIDGE)
                label.config(bg='black', fg='white')  
                label.config(font=labelfont)        
                label.grid(row = r, column = c)
                    #label.pack()
                #w=20
                    
                #c += 1
                r += 1


        print("DONE")

        fpt.close()
        #fp.close()
        fn.close()



###################################################################################################################

#   *******************************************
#Find most commond words from all the tweets
#
    def mostCommonWords(self):
        # most common words
        rawFile = open("auto_domain_tweets.csv")
        proDat = rawFile.read()
        count = Counter(proDat.split(' '))
        most_common =  count.most_common(60)
        r = 40

#################################################################################################################

    def Ptag(self):
        window = tk.Toplevel(self)

        frequency = {}
        uFile = open('uni_tweets.csv', 'r')
        ub_tweet=open('ubTweets.csv', 'w')
        for line in uFile.readlines():
            ub_tweet.write(line)

        uFile.close()
        ub_tweet.close()


        bFile = open('bi_tweet.csv', 'r')
        ub_tweet=open('ubTweets.csv', 'a')
        ptags=open('ptag.csv','w')
        for line in bFile.readlines():
            ub_tweet.write(line)

        bFile.close()
        ub_tweet.close()

        #Display popular tags
        labelfont = ('times', 10)
        r=0
        c=0
        with open('ubTweets.csv') as infile:
            counts = collections.Counter(l.strip() for l in infile)
            label = tkinter.Label(window, width =20 , height = 2,text = "Polpular Tags", relief = tkinter.RIDGE)
            label.config(bg='blue', fg='white')  
            label.config(font=labelfont)
            label.grid(row = r, column = c)
        for line, count in counts.most_common():
            if count>=8 :
                r+=1
                #print (line, count)
                
                label = tkinter.Label(window, width =20 , height = 2,text = line, relief=tkinter.RIDGE)
                label.config(bg='black', fg='white')  
                label.config(font=labelfont)        
                label.grid(row = r, column = c)
                ptags.write(line+"\n")

        ptags.close()


        
        
##        window = tk.Toplevel(self)
##
##        frequency = {}
##        uFile = open('uni_tweets.csv', 'r')
##        ub_tweet=open('ubTweets.csv', 'w')
##        for line in uFile.readlines():
##            ub_tweet.write(line)
##
##        uFile.close()
##        ub_tweet.close()
##
##
##        bFile = open('bi_tweet.csv', 'r')
##        ub_tweet=open('ubTweets.csv', 'a')
##        ptags=open('ptag.csv','w')
##        for line in bFile.readlines():
##            ub_tweet.write(line)
##
##        bFile.close()
##        ub_tweet.close()
##
##        #Display popular tags
##        labelfont = ('times', 10)
##        r=0
##        c=0
##
##        shutil.copy2('ubTweets.csv', 'ubTweets_copy.csv')
##        
##        infile=open('ubTweets.csv','r')
##        counts = collections.Counter(l.strip() for l in infile)
##        label = tkinter.Label(window, width =20 , height = 2,text = "Popular Tags", relief = tkinter.RIDGE)
##        label.config(bg='blue', fg='white')  
##        label.config(font=labelfont)
##        label.grid(row = r, column = c)
##
##            
##
##        corpusFlag="false"
##        ubCopy=open('ubTweets_copy.csv','r')
##        for txt in ubCopy.readlines():
##                
##            uCor = open('unigram_corpus.csv', 'r')
##            for l1 in uCor.readlines():
##                if txt==l1:
##                    corpusFlag="true"
##                    print (corpusFlag)
##            
##            
##            bCor = open('bigram_corpus.csv', 'r')
##            for l1 in bCor.readlines():
##                if txt==l1:
##                    corpusFlag="true"
##                    print (corpusFlag)
##            
##
##            tCor = open('trigram_corpus.csv', 'r')
##            for l1 in uCor.readlines():
##                if txt==l1:
##                    corpusFlag="true"
##                    print (corpusFlag)
##            
##        for line, count in counts.most_common():
##            
##            
##            
##                    
##            if count>=2 and corpusFlag=="true":
##                r+=1
##                print (corpusFlag)
##                
##                label = tkinter.Label(window, width =20 , height = 2,text = line, relief = tkinter.RIDGE)
##                label.config(bg='black', fg='white')  
##                label.config(font=labelfont)        
##                label.grid(row = r, column = c)
##                ptags.write(line+"\n")
##
##        ptags.close()
##        uCor.close()
##        bCor.close()
##        tCor.close()
##        ubCopy.close()
##
    def tagTweets(self) :
        text=""
        #window = tk.Toplevel(self)
        keywords = set()
        with open('ptag.csv','r') as list_file:
            for line in list_file:
                if line.strip():
                    keywords.add(line.strip())

##        k=""
##        i=0
        with open('auto_domain_tweets.csv','r') as master_file:
            with open('tag_tweet.csv', 'w') as search_results:
                for line in master_file:
                    if set(line.split()[:-1]) & keywords:
##                        k=keywords[i]
##                        print (k)
##                        text=line+","+str(keywords)
                        search_results.write(line)
##                        i+=1
##                        t=line.split()
##                        if t==keywords:
##                            label = tkinter.Label(window, width =20 , height = 2,text = row, relief = tkinter.RIDGE)
##                            label.config(bg='black', fg='white')  
##                            label.config(font=labelfont)        
##                            label.grid(row = r, column = 0)
##                            r=r+1
##                            
##
##        labelfont = ('times', 10)
##        # open file
####        label1 = tk.Label(window,width = 40, height = 2,text = "Tweets, relief" , relief = tkinter.RIDGE)
####        
####        label1.place(x = 0, y = 0 )
##        
####        label2 = tk.Label(window, width =20 , height = 2,text = "Sentiment Polarity", relief = tkinter.RIDGE)
####        label2 .place(x = 0, y = 0)
##        with open("tag_tweet.csv","r") as filee:
##            #file1.seek(0)
##            reader = csv.reader(filee)
##           
##            r = 0
##            for col in reader:
##                c = 0
##                w=40
##                
##                for row in col:
##         
##                    label = tkinter.Label(window, width =w , height = 2,text = row, relief = tkinter.RIDGE)
##                    label.config(bg='black', fg='white')  
##                    label.config(font=labelfont)        
##                    label.grid(row = r, column = c)
##                    #label.pack()
##                    w=20
##                    
##                    c += 1
##                r += 1
##

###################################################################################################################################
##def quit(self):
##    import sys;print('Exit Mode')
##    sys.exit()

def main():
##    """
##        The main function. 
##        This function sets up the root Tkinter window.
##    """
    
    # initialize root frame and run the app
    global root
    global check,ss
    root = tk.Tk()
    #root.geometry("800x400+100+100")
    root.geometry("1300x700+0+0")
    app = TweetimentFrame(root)
    root.mainloop()  


def definition():
    global domainVar,tweet,flag
    
if __name__ == '__main__':
    
    main() 
