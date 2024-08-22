#data cleaning needed to purify any outliers in the data set
#1927-2020 will be th ecurrent historical data that will be used
#im predicting a lot of margin of error
import pandas as pd
import pandas_datareader.data as pdw
import ta 
from sklearn.preprocessing import StandardScaler

SP = pd.read_csv("SPX.csv")
SP["Date"] = pd.to_datetime(SP["Date"])
SP.set_index("Date", inplace=True)
SP.fillna(method="ffill", inplace=True)

#Feature Engineered indicators
SP["SMA_300"] = ta.trend.sma_indicator(SP["Close"], window=20)
SP["RSI"] = ta.momentum.rsi(SP["Close"], window=14)
SP["MACD"] = ta.trend.macd(SP["Close"])
SP["Volatility"] = SP["Close"].rolling(window=20).std()

#ecconomic indicators needed to predict the future economic expectations
#Sources: FRED(Federal Reserve Economic Database)
gdp = pdw.DataReader('GDP', 'fred', start="1927-12-30", end="2020-11-04")
unemployment = pdw.DataReader("UNRATE", "fred", start="1927-12-30", end="2020-11-04")
cpi = pdw.DataReader("CPIAUCSL", "fred", start="1927-12-30", end="2020-11-04")
interest_rate = pdw.DataReader("FEDFUNDS", 'fred', start="1927-12-30", end="2020-11-04")

combined = pd.DataFrame({
    "S&P 500 Price": SP["Close"],
    "SMA_300": SP["SMA_300"].resample("M").last(),
    "MACD": SP["MACD"].resample("M").last(),
    "RSI": SP["RSI"].resample("M").last(),
    "Volatility": SP["Volatility"].resample("M").last(),
    "GDP": gdp["GDP"],
    "CPI": cpi["CPIAUCSL"],
    "Unemployment": unemployment["UNRATE"],
    "Interest Rate": interest_rate["FEDFUNDS"]
})

combined["GDP_Growth"] = combined["GDP"].pct_change()
combined["YoY_Inflation"] = combined["CPI"].pct_change(12)
combined.fillna(method="ffill", inplace=True)
combined.dropna(inplace=True)
combined.drop(["GDP", "CPI"], axis=1, inplace=True)
combined.sort_index()

#to get the training and testing data
sts = StandardScaler()
features = combined.columns
df = pd.DataFrame(sts.fit_transform(combined), columns=features, index=combined.index)

#df.to_csv("S&P_economic_indicator.csv", index=True) already done

print(df)

#now..onto some machine learning
