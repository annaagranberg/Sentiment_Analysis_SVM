import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

loaded_data = np.load('encoded_data.npz')

# From Clean_Vectorize.py
train_x_0 = loaded_data['train_x_0']
train_y_0 = loaded_data['train_y_0']
test_x_0 = loaded_data['test_x_0']
test_y_0 = loaded_data['test_y_0']

train_x_1 = loaded_data['train_x_1']
train_y_1 = loaded_data['train_y_1']
test_x_1 = loaded_data['test_x_1']
test_y_1 = loaded_data['test_y_1']

# X är (4588, 300) där 4588 är antalet dokument och 300 är längden på vektorn som representerar ett dokument 
# Y är (4588, 1) och representerar integers (0, 1, 2) alltså typen av bias.

class SVM:
    
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=10):
       
        self.lr = learning_rate  # Learning rate for gradient descent
        self.lambda_param = lambda_param  # Regularization parameter
        self.n_iters = n_iters  # Number of iterations for training
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def fit(self, X, y):
        
        n_samples, n_features = X.shape  # Get the number of samples and features in the training data
        self.w = np.zeros((len(np.unique(y)), n_features))  # Initialize the weight matrix
        self.b = np.zeros(len(np.unique(y)))  # Initialize the bias vector

        for class_label in np.unique(y):
         # Set class labels to -1 for samples not belonging to the current class
            binary_labels = np.where(y == class_label, 1, -1)
            for _ in range(self.n_iters):
                for idx, x_i in enumerate(X):
                    # Compute the condition based on the hinge loss
                    condition = binary_labels[idx] * (np.dot(x_i, self.w[class_label].T) - self.b[class_label]) >= 1
                    if condition.all():
                        # If the condition holds for all samples, update weights with regularization term
                        self.w[class_label] = self.w[class_label] - self.lr * (2 * self.lambda_param * self.w[class_label])
                    else:
                        # If the condition is not met, update weights and bias to account for misclassifications
                        self.w[class_label] = self.w[class_label] - self.lr * (2 * self.lambda_param * self.w[class_label] - np.outer(binary_labels[idx], x_i)) 
                        self.b[class_label] = self.b[class_label] - self.lr * binary_labels[idx]


    def predict(self, X):
        return np.argmax(np.dot(X, self.w.T) - self.b, axis=1) # Return the index with the highest score


svm_model = SVM()
svm_model.fit(train_x_0, train_y_0)
predictions = svm_model.predict(test_x_0)
accuracy = np.mean(predictions == test_y_0) 
print(f"Accuracy: {accuracy * 100:.2f}%") 


svm_model = SVM()
svm_model.fit(train_x_1, train_y_1)
predictions = svm_model.predict(test_x_1)
accuracy = np.mean(predictions == test_y_1) 
print(f"Accuracy: {accuracy * 100:.2f}%")  