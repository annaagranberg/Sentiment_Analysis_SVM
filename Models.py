
from sklearn import utils
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

#--------------------------- From Clean_Vectorize.py ---------------------------#
loaded_data = np.load('encoded_data.npz')

train_x_0 = loaded_data['train_x_0']
train_y_0 = loaded_data['train_y_0']
test_x_0 = loaded_data['test_x_0']
test_y_0 = loaded_data['test_y_0']

train_x_1 = loaded_data['train_x_1']
train_y_1 = loaded_data['train_y_1']
test_x_1 = loaded_data['test_x_1']
test_y_1 = loaded_data['test_y_1']

#--------------------------- Calculate accuracy ---------------------------#
def acc(true, pred):
  acc = 0
  for x,y in zip(true,pred):
    if(x == y): acc += 1
  return acc/len(pred)

# -------- Naive Bayes Classifier --------
bayes_0 = GaussianNB()
bayes_1 = GaussianNB()
bayes_0.fit(train_x_0,train_y_0)
bayes_1.fit(train_x_1,train_y_1)

print("Naive Bayes Classifier")
print(acc(test_y_0,bayes_0.predict(test_x_0)))
print(acc(test_y_1,bayes_1.predict(test_x_1)))

# -------- Random Forest from sklearn --------
forest_0 = RandomForestClassifier(n_estimators=100)
forest_1 = RandomForestClassifier(n_estimators=100)
forest_0.fit(train_x_0,train_y_0)
forest_1.fit(train_x_1,train_y_1)

print("Random Forest Classifier")
print(acc(test_y_0,forest_0.predict(test_x_0)))
print(acc(test_y_1,forest_1.predict(test_x_1)))

# -------- Support Vector Machine from sklearn ---------
svc_1 = SVC()
svc_1.fit(train_x_1,train_y_1)
svc_0 = SVC()
svc_0.fit(train_x_0,train_y_0)

print("Support Vector Machine")
print(acc(test_y_1,svc_1.predict(test_x_1)))
print(acc(test_y_0,svc_0.predict(test_x_0)))