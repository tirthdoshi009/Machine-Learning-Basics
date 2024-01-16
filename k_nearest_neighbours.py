from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
## Get the data
data = load_breast_cancer()

## Feature Variables
print(data.feature_names)

## Target Variable
print(data.target_names)


X_Train, X_Test, Y_Train, Y_Test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.6)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_Train, Y_Train)


print(clf.score(X_Test, Y_Test) * 100)