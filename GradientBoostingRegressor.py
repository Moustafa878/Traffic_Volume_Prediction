import math
from statistics import mean
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

data_df_train=pd.read_csv('Metro-Interstate-Traffic-Volume-Encoded.csv')
data_df_test=pd.read_csv('Metro-Interstate-Traffic-Volume-Encoded - test.csv')
x_train=data_df_train.drop(['traffic_volume'],axis=1).values
y_train=data_df_train['traffic_volume'].values

x_test=data_df_test[['holiday',	'temp',	'rain_1h','snow_1h','Year',	'Month','Day','Hour','weather_main']].values
gbreg = GradientBoostingRegressor(n_estimators=500, max_depth=10)
gbreg.fit(x_train, y_train)
pred=gbreg.predict(x_test)
#  evaluate the Regressor
pred_data=pd.DataFrame({'Actual':y_train,'Predicted':pred})
print(pred_data)
print(r2_score(y_train,pred))
p=[round(value) for value in pred]
print(p)
accuracy =accuracy_score(y_train,p)
print("accuracy  : ",accuracy)