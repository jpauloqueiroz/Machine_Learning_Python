import pandas as pd
import quandl

df = quandl.get("WIKI/GOOGL", api_key = "jeEUcYxisVGXxtCzUaNG")

df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100.0 # Volatility
df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0 # Daily percent change

df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

print(df.head)