# Jupyter notebook
import os
path_Data = 'E:/MSE/HomeWork/Python/Buoi1/OnClass/B/Stock Market/'

checkPath = os.path.isdir(path_Data)
print("The path is valid or not :", checkPath)  

checkFileApple = os.path.isfile(path_Data + "AAPL.csv")
checkFileGoogle = os.path.isfile(path_Data + "GOOG.csv")
checkFileMicrosoft = os.path.isfile(path_Data + "MSFT.csv")
print("The file is valid or not :", checkFileApple, checkFileGoogle, checkFileMicrosoft)

import pandas as pd

AAPL = pd.read_csv(path_Data + "AAPL.csv")
GOOG = pd.read_csv(path_Data + "GOOG.csv")
MSFT = pd.read_csv(path_Data + "MSFT.csv")

# print(type(AAPL), type(GOOG), type(MSFT))
# print("Shape AAPL : ", AAPL.shape, " Rows:", AAPL.shape[0], " Cols:", AAPL.shape[1])
# print("Shape GOOG : ", GOOG.shape, " Rows:", GOOG.shape[0], " Cols:", GOOG.shape[1])
# print("Shape MSFT : ", MSFT.shape, " Rows:", MSFT.shape[0], " Cols:", MSFT.shape[1])

AAPL.head()
GOOG.tail(10)
MSFT.head(3)
MSFT.iloc[2:4]
MSFT.loc[2:4]

AAPL['Adj Close'] = AAPL['Adj Close'].round(decimals=3)
AAPL.head()