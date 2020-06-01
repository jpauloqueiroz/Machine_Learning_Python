import pandas as pd
import quandl
import math

df = quandl.get("WIKI/GOOGL", api_key = "jeEUcYxisVGXxtCzUaNG")

df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100.0 # Volatility
df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0 # Daily percent change

df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

forecast_col = "Adj. Close"
df.fillna(-99999, inplace = True)

forecast_out = int(math.ceil(0.01 * len(df)))

df["Label"] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)

print(df.head())