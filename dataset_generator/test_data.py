#getting the testing data
#getting the data was a little hard
import pandas as pd
import ta
import wbdata as wb
import datetime #for dates
import pandas_datareader.data as pdw
from sklearn.preprocessing import StandardScaler


french_index = pd.read_csv("french_index.csv")
french_index["Date"]= pd.to_datetime(french_index["Date"])
french_index.set_index('Date', inplace=True)
french_index.fillna(method="ffill", inplace=True)
french_index.sort_index(inplace=True)
french_index.rename(columns={"Close":"CAC 40"}, inplace=True)
french_index = french_index[["CAC 40"]]


#getting the GDP values and defining the date range
start_date = datetime.datetime(1990, 3, 1)
end_date = datetime.datetime(2023, 12, 29)

indicators = {
    'NY.GDP.MKTP.CD': 'GDP',
    'FP.CPI.TOTL.ZG': 'Inflation',
    'SL.UEM.TOTL.ZS': 'Unemployment',
    'FR.INR.RINR': 'Interest rate'
}

economic_dta = wb.get_dataframe(indicators, country='FR', data_date=(start_date, end_date))
economic_dta.index = pd.to_datetime(economic_dta.index)
economic_dta = economic_dta.resample("D").ffill()
economic_dta = pd.DataFrame(economic_dta)
combined = french_index.join(economic_dta)
combined.fillna(0, inplace=True)


combined["SMA 200"] = ta.trend.sma_indicator(combined["CAC 40"], window=200)
combined["MACD"] =  ta.trend.macd(combined["CAC 40"])
combined["RSI"] = ta.momentum.rsi(combined["CAC 40"])
combined["Volatility"] = combined["CAC 40"].rolling(window=20).std()


combined["GDP_Growth"] = combined["GDP"].pct_change()
combined["YoY_Inflation"] = combined["Inflation"].pct_change(12)
combined.ffill(0, inplace=True)
combined.sort_index(inplace=True)
combined.drop(["GDP", "Inflation"], axis=1, inplace=True)
combined = combined.dropna(subset=["SMA 200", "MACD", "RSI", "Volatility", "GDP_Growth", "YoY_Inflation"])
combined = combined.fillna(combined.mean())
df = combined.interpolate()

#turning numerical to scalar values
sts = StandardScaler()
features = combined.columns
df = pd.DataFrame(sts.fit_transform(combined), columns=features, index=combined.index)

#df.to_csv("test_data.csv") #making it into a csv file

#data cleaning both training and testing data

df_1 = pd.read_csv("Training_data.csv")
df_2 = pd.read_csv("Testing_data.csv")

#both needs a placeholder to predict values
df_1["CAC 40 Price"] = None
df_2["S&P 500 Price"] = None

column_order_training = ['S&P 500 Price', 'SMA_300', 'MACD', 'RSI', 'Volatility', 'Unemployment', 'Interest Rate', 'GDP_Growth', 'YoY_Inflation']
column_order_testing = ['CAC 40 Price', 'SMA_300', 'MACD', 'RSI', 'Volatility', 'Unemployment', 'Interest Rate', 'GDP_Growth', 'YoY_Inflation']

df_1 = df_1[column_order_training]
df_2 = df_2[column_order_testing]
df_1 = df_1.fillna(0)
df_2 = df_2.fillna(0)

df_1.to_csv("training_data.csv", index=False)
df_2.to_csv("testing_data.csv", index=False)
