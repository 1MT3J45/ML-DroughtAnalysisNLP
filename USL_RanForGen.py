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

def plot(machine, X_input, y_input):
    classifier = machine
    X_train = X_input
    y_train = y_input

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    print(X_train.T.__len__())
    print(X_train.view())
    #X_test = sc_X.transform(X_test)

    # Visualising the Training set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.1),
                         np.arange(start=X_set[:, X_set.T.__len__()].min() - 1, stop=X_set[:, X_set.T.__len__()].max()+1
                                   , step=0.1))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, X_set.T.__len__()],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('Random forest classification (Train set)')
    plt.xlabel('Parameters')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
