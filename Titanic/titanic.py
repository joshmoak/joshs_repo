import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train = pd.get_dummies(train, columns=['Sex', 'Embarked'])
test = pd.get_dummies(test, columns=['Sex', 'Embarked'])

train.drop(["PassengerId",'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
test.drop(["PassengerId",'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

cols = list(train.columns)
cols.remove("Survived")
cats = ["Pclass", "Sex", "SibSp", "Parch", "Cabin", "Embarked"]

X = train[cols]
y = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)


gbm = lgb.LGBMClassifier()
gbm.fit(X_train, y_train)

print(gbm)