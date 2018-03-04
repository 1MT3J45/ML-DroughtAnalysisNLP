# Natural Language Processing

# Importing Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def processor(dataset_name):
    # Importing the Dataset
    #dataset = pd.read_csv("", delimiter='\t', quoting=3)
    dataset = dataset_name
    # We can have Double Quotes, Commas but Tab is best! also we are ignoring double quotes

    # Cleaning the Texts
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    # Creating a Corpus
    corpus = []
    for i in range(0, dataset.__len__()):
        review = re.sub('[^a-zA-Z]', ' ', str(dataset['tweet'][i]))
        review = review.lower()
        review = review.split()

        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)

    # Creating the Bag Of Words Model
    from sklearn.feature_extraction.text import CountVectorizer
    # Being the shape of 226, 145, max features is set to 144
    cv = CountVectorizer(max_features=144)

    # Sparse Matrix -> CV
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    # Fitting Logistic Regression to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X, y)
    return classifier, X, y


def usl_prediction(machine, X_input):
    classifier = machine
    X_data = X_input
    # Predicting the Test set results
    y_pred = classifier.predict(X_data)

    X_data = pd.DataFrame(X_data)
    y_pred = pd.DataFrame(y_pred)
    df = pd.concat([X_data, y_pred], axis=1)
    return df
