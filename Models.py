
from Clean_Vectorize import train_x_0, train_y_0, test_x_0, test_y_0, train_x_1, train_y_1, test_x_1, test_y_1 

import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from gensim.models import Doc2Vec
import gensim
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import multiprocessing
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re

#----------------------------- MODELS -------------------------------#

# All runned with the two “variations” of Doc2Vec:
    # Distributed memory (PV-DM)- inspired by the original Word2Vec algorithm
    # Distributed bag of words (PV-DBOW)- often works best on shorter texts
    
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
# 0.9197907585004359 with DBOW
# 0.6120313862249346 with DM 

# -------- Random Forest Classifer --------

# Create random forests with 100 decision trees
forest_0 = RandomForestClassifier(n_estimators=100)
forest_1 = RandomForestClassifier(n_estimators=100)

forest_0.fit(train_x_0,train_y_0)
forest_1.fit(train_x_1,train_y_1)
print("Random Forest Classifier")
print(acc(test_y_0,forest_0.predict(test_x_0)))
print(acc(test_y_1,forest_1.predict(test_x_1)))
# 0.9197907585004359 with DBOW
# 0.8108108108108109 with DM 

# -------- Support Vector Machine from sklearn ---------

svc_0 = SVC()
svc_1 = SVC()

svc_0.fit(train_x_0,train_y_0)
svc_1.fit(train_x_1,train_y_1)
print("Support Vector Machine")
print(acc(test_y_0,svc_0.predict(test_x_0)))
print(acc(test_y_1,svc_1.predict(test_x_1)))
# 0.946817785527463 with DBOW
# 0.8918918918918919 with DM 