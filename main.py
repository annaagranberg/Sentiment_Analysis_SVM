
# Tutorial by Medium : "Detecting political bias in online articles using NLP and classification models"

# Import the stopwords
from content.StopWords import _stopwords

#Deep Learning libraries
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

#Graphing libraries
import matplotlib.pyplot as plt
import seaborn as sns

#NLP libraries
import nltk
from gensim.models import Doc2Vec
import gensim
from gensim.models.doc2vec import TaggedDocument

#Machine learning libraries
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#Helper libraries
import multiprocessing
import numpy as np
import pandas as pd
#import mathpip ?? Finns inte?
from bs4 import BeautifulSoup
import re

# nltk.download('punkt') (Run once?)

# Load and shuffle the data
dataset = pd.read_csv('content/DataSet.csv', header=0) # Load the dataset
dataset = dataset.drop('url',axis=1) # Drop the URL column
dataset = dataset.iloc[np.random.permutation(len(dataset))] # Shuffle the dataset (Randomize the rows)

# The bias column is marked with letters, we need to replace them with numbers
dataset['bias'] = dataset['bias'].replace(['S','N','V'],[0,1,2]) # Replace the bias labels with numbers

# Graphical representation of the dataset
#bias_vals = dataset['bias'].value_counts()
#plt.figure()
#sns.barplot(x=bias_vals.index, y=bias_vals.values)
#plt.show()

# Clean the data; Remove uneccessary characters and stopwords, convert everything to lowercase
def clean(text):
    text = BeautifulSoup(text, features="lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = text.replace('„','')
    text = text.replace('“','')
    text = text.replace('"','')
    text = text.replace('\'','')
    text = text.replace('-','')
    text = text.lower()
    return text

def remove_stopwords(content):
    for word in _stopwords:
        content = content.replace(' '+word+' ',' ')
    return content

dataset['content'] = dataset['content'].apply(clean)
dataset['content'] = dataset['content'].apply(remove_stopwords)

# Split the dataset into training and testing data
train, test = train_test_split(dataset, test_size=0.2) # 80% training, 20% testing

# Encode the documents of text into vectors.

# To use Doc2Vec, a tagged document needs to be constructed using 
# tokenized text from the articles with the NLTK library.

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 3:
                continue
            tokens.append(word.lower())
    return tokens

train_tagged = train.apply(
   lambda r: TaggedDocument(words=tokenize_text(r['content']), tags=  [r.bias]), axis=1)
test_tagged = test.apply(
   lambda r: TaggedDocument(words=tokenize_text(r['content']), tags=[r.bias]), axis=1)

# There are two “variations” of Doc2Vec:
    # Distributed memory (PV-DM)- inspired by the original Word2Vec algorithm
    # Distributed bag of words (PV-DBOW)- often works best on shorter texts
    
cores = multiprocessing.cpu_count()
models = [
    # PV-DBOW 
    Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, sample=0, min_count=2, workers=cores),
    # PV-DM
    Doc2Vec(dm=1, vector_size=300, negative=5, hs=0, sample=0,    min_count=2, workers=cores)
]

# Build the vocabulary, train, and save the models.
for model in models:
  model.build_vocab(train_tagged.values)
  model.train(utils.shuffle(train_tagged.values),
    total_examples=len(train_tagged.values),epochs=30)

models[0].save("doc2vec_articles_0.model")
models[1].save("doc2vec_articles_1.model")

# Using the trained models, encode the text from the articles into vectors of length 300.
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    classes, features = zip(*[(doc.tags[0],
      model.infer_vector(doc.words, epochs=20)) for doc in sents])
    return features, classes

# PV_DBOW encoded text
train_x_0, train_y_0 = vec_for_learning(models[0], train_tagged)
test_x_0, test_y_0 = vec_for_learning(models[0], test_tagged)
# PV_DM encoded text
train_x_1, train_y_1 = vec_for_learning(models[1], train_tagged)
test_x_1, test_y_1 = vec_for_learning(models[1], test_tagged)


#----------------------------- MODELS -------------------------------#

# -------- Naive Bayes Classifier --------
bayes_0 = GaussianNB()
bayes_1 = GaussianNB()

bayes_0.fit(train_x_0,train_y_0)
bayes_1.fit(train_x_1,train_y_1)
#Helper function for calculating accuracy on the test set.
def acc(true, pred):
  acc = 0
  for x,y in zip(true,pred):
    if(x == y): acc += 1
  return acc/len(pred)
print("Naive Bayes Classifier")
print(acc(test_y_0,bayes_0.predict(test_x_0)))
print(acc(test_y_1,bayes_1.predict(test_x_1)))
# 0.9197907585004359
# 0.6120313862249346 

# -------- Random Forest Classifer --------

# Create random forests with 100 decision trees
forest_0 = RandomForestClassifier(n_estimators=100)
forest_1 = RandomForestClassifier(n_estimators=100)

forest_0.fit(train_x_0,train_y_0)
forest_1.fit(train_x_1,train_y_1)
print("Random Forest Classifier")
print(acc(test_y_0,forest_0.predict(test_x_0)))
print(acc(test_y_1,forest_1.predict(test_x_1)))
# 0.9197907585004359
# 0.8108108108108109

# -------- Support Vector Machine ---------

svc_0 = SVC()
svc_1 = SVC()

svc_0.fit(train_x_0,train_y_0)
svc_1.fit(train_x_1,train_y_1)
print("Support Vector Machine")
print(acc(test_y_0,svc_0.predict(test_x_0)))
print(acc(test_y_1,svc_1.predict(test_x_1)))
# 0.946817785527463
# 0.8918918918918919