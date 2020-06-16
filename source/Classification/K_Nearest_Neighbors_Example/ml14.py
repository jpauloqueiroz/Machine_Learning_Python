import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd 

df = pd.read_csv(r"C:\Users\Joao\Code\Machine_learning_python\source\Classification\K_Nearest_Neighbors_Example\breast-cancer-wisconsin.data.txt")
df.replace("?", -99999, inplace = True)
df.drop(["id"], 1, inplace = True)

X = np.array(df.drop(["class"], 1))
y = np.array(df["class"])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

# Predicting
example = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
example = example.reshape(2, -1)

prediction = clf.predict(example)
print(prediction)