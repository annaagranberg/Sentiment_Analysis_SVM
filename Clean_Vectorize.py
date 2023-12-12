
from content.StopWords import _stopwords

from sklearn import utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from gensim.models import Doc2Vec
import gensim
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re

# nltk.download('punkt') // Only needed for the first time

dataset = pd.read_csv('content/DataSet.csv', header=0)
dataset = dataset.drop('url',axis=1) # Drop the URLs
dataset = dataset.iloc[np.random.permutation(len(dataset))] # Shuffle the dataset
dataset['bias'] = dataset['bias'].replace(['S','N','V'],[0,1,2]) # Replace the bias labels with numbers

# Graphical representation of the dataset
#bias_vals = dataset['bias'].value_counts()
#plt.figure()
#sns.barplot(x=bias_vals.index, y=bias_vals.values)
#plt.show()

# Remove uneccessary characters and stopwords, convert everything to lowercase
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

train, test = train_test_split(dataset, test_size=0.2) # 80% training, 20% testing

# Encode the documents of text into vectors.

# To use Doc2Vec, a tagged document needs to be constructed using 
# Tokenize with the NLTK library.

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


# Distributed memory (PV-DM)
# Distributed bag of words (PV-DBOW)
    
cores = multiprocessing.cpu_count()
models = [
    # PV-DBOW 
    Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, sample=0, min_count=2, workers=cores),
    # PV-DM
    Doc2Vec(dm=1, vector_size=300, negative=5, hs=0, sample=0,    min_count=2, workers=cores)
]

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

np.savez('encoded_data.npz', train_x_0=train_x_0, train_y_0=train_y_0, test_x_0=test_x_0, test_y_0=test_y_0,
         train_x_1=train_x_1, train_y_1=train_y_1, test_x_1=test_x_1, test_y_1=test_y_1)


