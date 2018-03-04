# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_fit(data_frame):
    # IMPORTING DATASET
    dataset = data_frame

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
    cv = CountVectorizer(max_features=144)

    # Sparse Matrix -> CV
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    # Splitting Data into Training & Testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # Fitting Random Forest class to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    return classifier, X_test, y_test


def rfg_spv_predict(machine, X_input, y_input):
    X_test = X_input
    y_test = y_input
    classifier = machine
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm
