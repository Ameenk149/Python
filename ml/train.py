import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

is_training = False

dataset = pd.read_csv("dataset/connect4.csv")

X = dataset.iloc[:,0:-1]
y = dataset[["result"]]

X = X.replace("x", 1)
X = X.replace("b", 0)
X = X.replace("o", -1)

y = y.replace("win", 1)
y = y.replace("draw", 0)
y = y.replace("loss", -1)

print("Dataset (Board):")
print(X)

print("Dataset (Game Results):")
print(y)

if is_training:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    model = RandomForestClassifier(max_depth=50, n_estimators=100, verbose=True)
    model.fit(X_train, y_train.values.ravel().astype("int"))

    print("Test Predicting:")
    print(X_test.values)

    test_result = model.predict(X_test.values)

    print("Result:")
    print(test_result)

    print("Accuracy: "+str(accuracy_score(y_test.values.ravel().astype("int"), test_result)*100.00) + "%")

    # 83% Accuracy

else:
    print("Training...")
    model = RandomForestClassifier(max_depth=50, n_estimators=100)
    model.fit(X.values, y.values.ravel().astype("int"))
    pickle.dump(model, open("connect4_model_84accuracy", 'wb'))
    print("Saved.")