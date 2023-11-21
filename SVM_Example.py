
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import time
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

X, Y = make_blobs(n_samples = 40,centers=2, cluster_std=1.2,n_features=2,random_state=42)

#convert to -1's and 1's
for i,j in enumerate(Y):
    if j == 0:
        Y[i] = -1
    elif j == 1:
        Y[i] = 1
        
#group for plotting
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=Y))
names = {-1:'Group 1', 1:'Group 2'}
colors = {-1:(0,100/255,0,0.9), 1:(138/255,43/255,226/255,0.9)}
grouped = df.groupby('label')

#Example Lines: 2 points needed per line
ex_line_x1 = np.linspace(-4,6,100)
ex_line_y1 = 1*ex_line_x1+4
ex_line_x2 = np.linspace(-4,6,100)
ex_line_y2 = 0.2*ex_line_x2+4.5
ex_line_x3 = np.linspace(-4,6,100)
ex_line_y3 = -0.1*ex_line_x3+5.5

#plot settings and labels
fig = plt.figure(figsize=(13,9))
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Two Clusters of Data", fontsize=20)
ax.set_xlabel("Feature 1", fontsize=18)
ax.set_ylabel("Feature 2", fontsize=18)
ax.set_facecolor((245/255,245/255,245/255))

#plot the data and example line
for key, group in grouped:
    ax.scatter(group.x,group.y, label=names[key], color=colors[key],edgecolor=(0,0,0,0.75),s=350)
ax.plot(ex_line_x1, ex_line_y1, color=(0.95,0.1,0.2,0.8), label='Line 1',linewidth=4)
ax.plot(ex_line_x2, ex_line_y2, color=(0.1,0.3,0.95,0.8), label='Line 2',linewidth=4)
ax.plot(ex_line_x3, ex_line_y3, color=(0.1,0.7,0.8,0.8), label='Line 3',linewidth=4)
ax.legend(markerscale=1,fontsize="x-large")
plt.show()

#prepare datasets
#test sets
x_test = X[20:]
x_test = np.c_[x_test,np.ones(len(x_test))]
y_test = Y[20:]

#training sets
x = X[:20]
y = Y[:20]
#group for plotting
df_train = pd.DataFrame(dict(x=x[:,0], y=x[:,1], label=y))
grouped_train = df_train.groupby('label')

#add bias to sample vectors
x = np.c_[x,np.ones(len(x))]

#initialize weight vector
w = np.zeros(len(x[0]))

#learning rate 
lam = 0.001
#array of number for shuffling
order = np.arange(0,len(x),1)

margin_current = 0
margin_previous = -10

pos_support_vectors = 0
neg_support_vectors = 0

not_converged = True
t =0 
start_time = time.time()

while(not_converged):
    margin_previous = margin_current
    t += 1
    pos_support_vectors = 0
    neg_support_vectors = 0
    
    eta = 1/(lam*t)
    fac = (1-(eta*lam))*w
    random.shuffle(order)
    for i in order:  
        prediction = np.dot(x[i],w)
        
        #check for support vectors
        if (round((prediction),1) == 1):
            pos_support_vectors += 1
            #pos support vec found
        if (round((prediction),1) == -1):
            neg_support_vectors += 1
            #neg support vec found
            
        #misclassification
        if (y[i]*prediction) < 1 :
            w = fac + eta*y[i]*x[i]            
        #correct classification
        else:
            w = fac
            
    if(t>10000):    
        margin_current = np.linalg.norm(w)
        if((pos_support_vectors > 0)and(neg_support_vectors > 0)and((margin_current - margin_previous) < 0.01)):
            not_converged = False

#print running time
print("--- %s seconds ---" % (time.time() - start_time))

#create grid to draw decision boundary
grid_res = 200
xline = np.linspace(min(X[:,0]-(0.5*np.std(X[:,0]))),max(X[:,0]+(0.5*np.std(X[:,0]))),grid_res)
yline = np.linspace(min(X[:,1]-(0.5*np.std(X[:,1]))),max(X[:,1]+(0.5*np.std(X[:,1]))),grid_res)
grid = []
gridy = []
for i in range(grid_res):
    for j in range(grid_res):
        grid.append([xline[i],yline[j]])
        if (np.dot(w,[xline[i],yline[j],1]))>1:
            gridy.append((138/255,43/255,226/255,0.1))
            #gridy.append('lightsteelblue')
        elif (np.dot(w,[xline[i],yline[j],1]))<-1:
            gridy.append((0,100/255,0,0.1))
            #gridy.append('steelblue')
        elif (round((np.dot(w,[xline[i],yline[j],1])),2) == 0):
            gridy.append((0,0,0,1))
        else:
            gridy.append((245/255,245/255,245/255))
            
grid = np.asarray(grid)
gridy = np.asarray(gridy)

#plot the data
fig = plt.figure(figsize=(15,11))
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Blobs Classified by SVM", fontsize=20)
ax.set_xlabel("Feature 1", fontsize=18)
ax.set_ylabel("Feature 2", fontsize=18)
ax.scatter(grid[:,0], grid[:, 1], marker='o',c=gridy,s=10)
for key, group in grouped_train:
    ax.scatter(group.x,group.y, label=names[key], color=colors[key],edgecolor=(0,0,0,0.75),s=350)
ax.legend(markerscale=1,fontsize=20,fancybox=True)
plt.show()

#test classifier on test set
y_pred = ([])
for i in x_test:
    pred = np.dot(w,i)
    if(pred > 0):
        {y_pred.append(1)}
    elif(pred < 0):
        y_pred.append(-1)
        

y_pred_labels =([])
for i,val in enumerate(y_pred):
    if(y_test[i] == y_pred[i]):
        y_pred_labels.append(1)
    else:
        y_pred_labels.append(0)
        
#group for plotting
df_test = pd.DataFrame(dict(x=x_test[:,0], y=x_test[:,1], pred=y_pred_labels,label=y_test))
grouped_test = df_test.groupby('label')
grouped_pred = df_test.groupby('pred') 
pred_colors = {1:'lime', 0:'red'}
pred_names = {1:'correct',0:'incorrect'}

#plot decision grid with prediction values
fig = plt.figure(figsize=(15,11))
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Blobs Classified by SVM", fontsize=20)
ax.set_xlabel("Feature 1", fontsize=18)
ax.set_ylabel("Feature 2", fontsize=18)
ax.scatter(grid[:,0], grid[:, 1], marker='o',c=gridy,s=10)
for key, group in grouped_test:
    ax.scatter(group.x,group.y, label=names[key], color=colors[key],edgecolor=(0,0,0,0),s=350)
for key, group in grouped_pred:
    ax.scatter(group.x,group.y,label=pred_names[key],color=(0,0,0,0),linewidth=2,edgecolor=pred_colors[key],s=350)
ax.legend(markerscale=1,fontsize=20,fancybox=True)
plt.show()