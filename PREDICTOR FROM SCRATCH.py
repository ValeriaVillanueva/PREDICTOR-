#!/usr/bin/env python
# coding: utf-8

# In[141]:


import pandas as pd

path = r"C:\Users\valef\Downloads\Indicadores_municipales_sabana_DA (1).csv"

df = pd.read_csv(path, encoding='latin-1')

print(df)


# In[142]:


# Print the count of missing values in each column
print(df.isnull().sum())


# In[143]:


#Replace the missing values with the mean of the column
df = df.fillna(df.mean(numeric_only=True))


# In[144]:


# Print the count of missing values in each column 
print(df.isnull().sum())


# In[145]:


# Remove specified columns from the DataFrame 'df'
df = df.drop(columns = ["nom_ent", "mun", "clave_mun", "nom_mun","gdo_rezsoc00", "gdo_rezsoc05", "gdo_rezsoc10","ent","plb"])


# In[146]:


# Create a new column 'Conversion of target' 
# The new column contains binary values (0 or 1) based on the following condition:
# If the number of 'N_plb' is greater than half of the population in 'pobtot_ajustada',
# set the value to 1 (True); otherwise, set it to 0 (False).

df['Conversion of target'] = (df['N_plb'] > df['pobtot_ajustada'] / 2).astype(int)



# In[147]:


df


# # KNN NO LIBRARIES

# The code first sets a random seed for reproducibility and shuffles the DataFrame 'df' by randomly sampling the entire dataset. This shuffling ensures data is not ordered in a specific way, avoiding potential bias during training. Subsequently, the data is split into training (80%) and testing (20%) sets. Features (X_train and X_test) are separated from labels (y_train and y_test).

# In[154]:


import numpy as np

np.random.seed(0)

df = df.sample(frac=1)

# Divide dataset
train_size = int(0.8 * len(df))
train_set = df[:train_size]
test_set = df[train_size:]

# Separation
X_train = train_set.iloc[:, :-1].values
y_train = train_set.iloc[:, -1].values
X_test = test_set.iloc[:, :-1].values
y_test = test_set.iloc[:, -1].values


# The mean is calculated for X_train along the feature dimensions (axis=0), and the standard deviation is determined in a similar manner. Subsequently, the data is standardized by subtracting the mean and dividing by the standard deviation. This standardization process is applied to both the training data (X_train) and the test data (X_test).

# In[155]:


# Mean and standar deviation
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

# Normalization of data
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# The KNN function predicts using input features and labels. It calculates distances between each test point and all training data points, considering the Euclidean distance metric. The 'k' nearest neighbors are selected by sorting these distances. The prediction for a test point results from a majority vote among these k-nearest labels. The code then computes accuracy by comparing predicted labels (y_pred) to actual test labels (y_test) and prints the accuracy as a percentage.

# In[156]:


# Definir la función KNN
def knn(X_train, y_train, X_test, k):
    y_pred = []
    for test_point in X_test:
        distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_indices]
        pred = np.bincount(k_nearest_labels).argmax()
        y_pred.append(pred)
    return np.array(y_pred)

# Calcular la precisión
y_pred = knn(X_train, y_train, X_test, k=4)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"Precisión: {accuracy * 100:.2f}%")


# # PERCEPTRON NO LIBRARIES

# The code initializes the weights and bias for the Perceptron model. It assumes the presence of a DataFrame 'df' with features and labels, separating these into 'X' and 'y'. The Perceptron's 'step function' is defined for activation. The weights are randomly initialized using a specified random seed, with the bias set to a random value.

# In[157]:


import numpy as np

# Separation of features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Define the activation function of perceptron)
def step_function(z):
    return np.where(z >= 0, 1, 0)

# Initialize the weights and bias in a random form 
np.random.seed(0)
weights = np.random.rand(X.shape[1])  
bias = np.random.rand()  


# The code sets a learning rate of 0.1 and runs the Perceptron training for 100 epochs. It calculates the weighted sum ('z') from the dot product of 'X' and weights plus bias, applies the 'step function' for predictions, computes the error (y - predicted), and updates weights and bias scaled by the learning rate.

# In[158]:


#Set learning rate
learning_rate = 0.1

# Define epochs
epochs = 100

# Train perceptron
for epoch in range(epochs):
    # Calculate the weighted sum plus bias (z)
    z = np.dot(X, weights) + bias

    # Apply the activation function
    predicted = step_function(z)

    # Calculate the error
    error = y - predicted

    # Update the weights and bias
    weights += learning_rate * np.dot(X.T, error)
    bias += learning_rate * np.sum(error)


# Predictions are derived by applying the 'step function' to the weighted sum, computed as the dot product of feature matrix 'X' and the weight vector with added bias. The code assesses accuracy by comparing these predictions to actual labels (y), computing mean accuracy as a percentage, and printing the result to the console.

# In[159]:


# Develop predictiona
predictions = step_function(np.dot(X, weights) + bias)

# Calculate the accuracy
accuracy = np.mean(predictions == y) * 100
print(f"Precisión: {accuracy:.2f}%")


# # PERCEPTRON 

# This code utilizes scikit-learn to train a Perceptron classifier for a binary classification task. It begins by separating features and labels, splitting the data into training and testing sets. The Perceptron model is then trained with specified parameters, and predictions are made on the test data. The code computes and displays the accuracy of the model's predictions.

# In[150]:


import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Separate features (X) and labels (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Perceptron classifier
perceptron = Perceptron(max_iter=1000, eta0=0.1, random_state=0)

# Train the Perceptron on the training set
perceptron.fit(X_train, y_train)

# Make predictions on the test set
y_pred = perceptron.predict(X_test)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión: {accuracy * 100:.2f}%")



# # KNN

# 
# This code uses the K-Nearest Neighbors (KNN) algorithm from scikit-learn to classify data. It begins by separating features (X) and labels (y) from a DataFrame 'df' and splitting the data into training and testing sets. A KNN classifier is created with two nearest neighbors and trained on the training data. Predictions are made on the test data, and the code calculates and prints the accuracy score, indicating how well the model predicts the labels.

# In[105]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Separate features (X) and labels (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a k-NN classifier
knn = KNeighborsClassifier(n_neighbors=2)  

# Train the k-NN model on the training set
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión: {accuracy * 100:.2f}%")

