# mport important packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib


# path to the dataset
filename = "../data/breast-cancer-wisconsin.data"

# load data
data = pd.read_csv(filename)

# replace "?" with -99999
data = data.replace("?", -99999)

# drop id column
data = data.drop(["id"], axis=1)

# Define X (independent variables) and y (target variable)

X = data.drop(["class"], axis=1)
y = data["class"]

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# call our classifer and fit to our data
classifier = KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    algorithm="auto",
    leaf_size=25,
    p=1,
    metric="minkowski",
    n_jobs=-1,
)
# training the classifier
classifier.fit(X_train, y_train)

# test our classifier
result = classifier.score(X_test, y_test)
print("Accuracy score is. {:.1f}".format(result))

# save our classifier in the model directory
model_name = "KNN_classifier"
joblib.dump(classifier, "../models/{}.pkl".format(model_name))
