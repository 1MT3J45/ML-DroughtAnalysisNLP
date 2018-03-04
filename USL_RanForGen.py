# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_fit(data_frame):
    # Importing Dataset
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

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    # Fitting Random Forest class to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X, y)
    return classifier, X, y

def rfg_usp_predict(machine, X_input):
    X_ip = X_input
    classifier = machine

    y_pred = classifier.predict(X_ip)

    X_data = pd.DataFrame(X_ip)
    y_pred = pd.DataFrame(y_pred)
    df = pd.concat([X_data, y_pred], axis=1)

    return df
