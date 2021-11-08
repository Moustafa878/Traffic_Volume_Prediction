import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeRegressor

data_df_train=pd.read_csv('Metro-Interstate-Traffic-Volume-Encoded.csv',)
data_df_test=pd.read_csv('Metro-Interstate-Traffic-Volume-Encoded - test.csv')

x_train=data_df_train.drop(['traffic_volume'],axis=1).values
y_train=data_df_train['traffic_volume'].values
'''
print(type(x_train))
print(len(x_train))
print(len(y_train)) '''

x_test=data_df_test[['holiday',	'temp',	'rain_1h','snow_1h','Year',	'Month','Day','Hour','weather_main']].values
#print(len(x_test))
ml=RandomForestRegressor(max_depth=12, min_samples_split=5, n_estimators=50)
ml.fit(x_train,y_train)


pred=ml.predict(x_test)


#print(math.sqrt(mean_squared_error(y_test,pred)))
#print(pred)
pred_data=pd.DataFrame({'Actual':y_train,'Predicted':pred})
print(pred_data)
print('R-square  :',r2_score(y_train,pred))
t=[round(value) for value in y_train]
p=[round(value) for value in pred]

#print(p)
accuracy =accuracy_score(t,p)
