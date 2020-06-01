import pandas as pd
import quandl, math
import numpy as np 
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression

df = quandl.get("WIKI/GOOGL", api_key = "jeEUcYxisVGXxtCzUaNG")

df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100.0 # Volatility
df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0 # Daily percent change

df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

forecast_col = "Adj. Close"
df.fillna(-99999, inplace = True)

forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)

df["Label"] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)

X = np.array(df.drop(["Label"], 1))
y = np.array(df["Label"])
X = preprocessing.scale(X)
y = np.array(df["Label"])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs = -1) # Algorithm
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)


print(accuracy)