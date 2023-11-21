import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

loaded_data = np.load('encoded_data.npz')

train_x_0 = loaded_data['train_x_0']
train_y_0 = loaded_data['train_y_0']
test_x_0 = loaded_data['test_x_0']
test_y_0 = loaded_data['test_y_0']

train_x_1 = loaded_data['train_x_1']
train_y_1 = loaded_data['train_y_1']
test_x_1 = loaded_data['test_x_1']
test_y_1 = loaded_data['test_y_1']

# Check the dimensions of the data
print(len(test_x_0), len(test_y_0))
print(test_x_0.shape, test_y_0.shape)

# Slutsatser: 

# X är (4588, 300) där 4588 är antalet dokument och 300 är längden på vektorn som representerar ett dokument 
# Y är (4588, 1) och representerar integers (0, 1, 2) alltså typen av bias.

# print(train_y_0[0:10]) -> [1,1,1,2,1,0,0,2,0,0]


#----------------- Plotting with PCA -----------------#

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

x_data = test_x_0
y_data = test_y_0

# Apply PCA to reduce x_data to 3 dimensions
pca = PCA(n_components=3)
x_3d = pca.fit_transform(x_data)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x_3d[:, 0], x_3d[:, 1], x_3d[:, 2], c=y_data, cmap='viridis')

# Add labels and legend
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D Scatter Plot with PCA')

# Add colorbar for the class labels
cbar = plt.colorbar(scatter)
cbar.set_label('Class Label')

# Show the plot
plt.show()

#------------------------------------------------------#

#----------------- Support vector machine -----------------#

# Simple linear support vector machine

import numpy as np

class SVM:
    
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=10):
        # Initialization method for the SVM class
        # It sets up the hyperparameters and attributes of the SVM model
        self.lr = learning_rate  # Learning rate for gradient descent
        self.lambda_param = lambda_param  # Regularization parameter
        self.n_iters = n_iters  # Number of iterations for training
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def fit(self, X, y):
        # Training method for the SVM model
        # X: Feature matrix (training data)
        # y: Vector of labels

        n_samples, n_features = X.shape  # Get the number of samples and features in the training data
        self.w = np.zeros((len(np.unique(y)), n_features))  # Initialize the weight matrix
        self.b = np.zeros(len(np.unique(y)))  # Initialize the bias vector

        for _ in range(self.n_iters):
            # Iterating over the specified number of training iterations

            for idx, x_i in enumerate(X):
                # Iterating over each training sample

                # Compute the condition based on the hinge loss
                condition = y[idx] * (np.dot(x_i, self.w.T) - self.b) >= 1

                if condition.all():
                    # If the condition holds for all samples, update weights with regularization term
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # If the condition is not met, update weights and bias to account for misclassifications
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.outer(y[idx], x_i))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        # Prediction method for the SVM model
        # X: Feature matrix for making predictions

        # Return the index of the class with the highest score for each sample
        return np.argmax(np.dot(X, self.w.T) - self.b, axis=1)

# Initialize and train the SVM
svm_model = SVM()
svm_model.fit(train_x_0, train_y_0)

# Make predictions
predictions = svm_model.predict(test_x_0)

# Evaluate the model
accuracy = np.mean(predictions == test_y_0)  # Compute accuracy by comparing predictions with true labels
print(f"Accuracy: {accuracy * 100:.2f}%")  # Print the accuracy of the trained SVM model


#------------------------- PLOT -----------------------------#

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(test_x_0[:, 0], test_x_0[:, 1], test_x_0[:, 2], c=test_y_0, cmap='viridis', marker='o', label='Data Points')

# Plot the decision boundary (hyperplane)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()

xx, yy, zz = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10), np.linspace(zlim[0], zlim[1], 10))
xyz = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

Z = svm_model.predict(xyz)
Z = Z.reshape(xx.shape)

ax.contour3D(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Add labels and legend
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.legend()
plt.title('SVM Classification with Decision Boundary')

# Show the plot
plt.show()