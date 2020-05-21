# Import important packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
import yaml


# folder to load config file
CONFIG_PATH = "../config/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("my_config.yaml")


# load data
data = pd.read_csv(os.path.join(config["data_directory"], config["data_name"]))

# replace "?" with -99999
data = data.replace("?", -99999)

# drop id column
data = data.drop(config["drop_columns"], axis=1)

# Define X (independent variables) and y (target variable)
X = np.array(data.drop(config["target_name"], 1))
y = np.array(data[config["target_name"]])

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config["test_size"], random_state=42
)


# call our classifer and fit to our data
classifier = KNeighborsClassifier(
    n_neighbors=config["n_neighbors"],
    weights=config["weights"],
    algorithm=config["algorithm"],
    leaf_size=config["leaf_size"],
    p=config["p"],
    metric=config["metric"],
    n_jobs=config["n_jobs"],
)
# training the classifier
classifier.fit(X_train, y_train)

# test our classifier
result = classifier.score(X_test, y_test)
print("Accuracy score is. {:.1f}".format(result))

# save our classifier in the model directory
joblib.dump(classifier, os.path.join(config["model_directory"], config["model_name"]))
