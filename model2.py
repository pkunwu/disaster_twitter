import spacy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from load import load
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import spacy


filepath = r'..\twitter_disaster\data\train_cleaned.csv'

df = load(filepath)

nlp = spacy.load('en_core_web_lg')

with nlp.disable_pipes():
    vectors = np.array([nlp(text).vector for text in  df['text_cleaned']])

np.savetxt('../twitter_disaster/data/word2vec.csv', vectors, delimiter=",")

X_train, X_test, y_train, y_test = train_test_split(vectors, df['target_relabeled'],test_size=0.1, random_state=1)
svc = LinearSVC(random_state=1, dual=False, max_iter=10000)
svc.fit(X_train, y_train)
print(f"Accuracy: {svc.score(X_test, y_test) * 100:.3f}%", )
