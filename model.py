import pickle
import pandas as pd
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB ,ComplementNB , BernoulliNB

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


Fake = pd.read_csv('Fake.csv')
Real = pd.read_csv('True.csv')

Fake['Label']=0
Real['Label']=1

Dataset = Real.append(Fake, ignore_index=True)
Dataset= Dataset.fillna("")



# Train the model

TFIDF = TfidfVectorizer(stop_words = "english")
X_train_tf = TFIDF.fit_transform(Dataset['text'])

NaiveBayes = BernoulliNB()
NaiveBayes.fit(X_train_tf, Dataset['Label'])


pickle.dump(NaiveBayes,open("model.pkl","wb"))

