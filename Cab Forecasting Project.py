import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os as osf
from pandas_profiling import ProfileReport
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score


#foldername = osf.getcwd() + 'D:\\NIT\\Predictive Analytics\\datasets'
foldername = 'D:\\NIT\\Predictive Analytics\datasets\\'
dftest=pd.read_csv(foldername +'\\test.csv')
dftrain=pd.read_csv(foldername + '\\train.csv')
testlabels=pd.read_csv(foldername + '\\test_label.csv', header=None)
trainlabels=pd.read_csv(foldername + '\\train_label.csv', header=None)
testlabels.columns=['bookings']
trainlabels.columns=['bookings']
testlabels.shape
trainlabels.shape

testlabels.head()
trainlabels.head()
dftest.head()
dftrain
test=pd.concat([dftest,testlabels],axis=1)
test
test.info()
train=pd.concat([dftrain,trainlabels],axis=1)
train
train.info()
test.reset_index(drop=True, inplace=True)
test
train.reset_index(drop=True, inplace=True)


test.head()
test.shape
pd.set_option('display.max_columns',100)
test

df=pd.concat([test,train])
df
df.count()
df.describe()
type(df)

df.corr()###atemp and temp shows high correlation hence one of the variables can eb dropped
df.head(10)

###creating additional features

pd.to_datetime(df['datetime'])
df['Date']=pd.to_datetime(df['datetime']).dt.date
df['Time']=pd.to_datetime(df['datetime']).dt.time
df.info()
df['Month']=pd.to_datetime(df['datetime']).dt.month
df['Weekday']=pd.to_datetime(df['datetime']).dt.weekday
df
help(datetime)

#####encoding the variables

df1=pd.get_dummies(df[['season','weather']])
df2=pd.concat((df,df1),axis=1)
df2.info()
df2.drop(['season','weather'],axis=1,inplace=True)
df2.info()

#### Dropping the irrelevant columns

df2.drop(['datetime'],axis=1,inplace=True)
df2.drop(['atemp'],axis=1, inplace=True)
df2
###changing the datatypes

#df2[['holiday','workingday']]=df2[['holiday','workingday']].astype(bool)
#df2.info()
#####Data visualization
##### Detecting outliers in various variables and performing log transformation

sns.boxplot(data=df2,x='temp',y='bookings',palette='CMRmap')
sns.boxplot(data=df2, x='bookings', palette='CMRmap')
sns.boxplot(data=df2, x='humidity',y='bookings', palette='CMRmap')
sns.boxplot(data=df2, x='windspeed',y='bookings', palette='CMRmap')

df2=df2.loc[df2.bookings<900]
df2

sns.boxplot(data=df2, x='bookings', palette='CMRmap')

sns.boxplot(data=df2, x='windspeed',y='bookings', palette='CMRmap')



####Profiling

profile=ProfileReport(df2,explorative=True)
profile.to_widgets()
profile.to_file("profile report.html")


#### spliting the set into train and test sets
X=df2[['temp','humidity','windspeed','holiday', 'workingday','season_Fall','season_Spring','season_Summer','season_Winter','weather_ Clear + Few clouds','weather_ Light Snow, Light Rain']]
#X=df2[['temp','humidity','windspeed', 'holiday', 'workingday']]

y=df2['bookings']
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=70, train_size=30, random_state=1,)

###from the above given statistics it is evident that there were a few outliers and 
### the values for bookings over 900 have been removed

#### Random Forest Regressor

forest_reg=RandomForestRegressor()
n_estimators=[10,15,20,25,30,35,40,45]
max_depth=[2,10]
max_features=['auto']

param_grid=[{'n_estimators': [5,15,25,33,45,60,80], 'max_depth' : [2,10], 'max_features' : ['auto'], 'min_samples_split': [2,5], 'min_samples_leaf': [1,2]}]

grid_search=GridSearchCV(forest_reg, param_grid,cv=5, verbose=3, n_jobs=4)

grid_search.fit(X_train,y_train)

grid_search.best_params_

forest_reg2=RandomForestRegressor(n_estimators=5, max_depth=10, max_features='auto', min_samples_leaf=2, min_samples_split=5)
forest_reg2.fit(X_train,y_train)
y_pred=forest_reg2.predict(X_test)

from math import sqrt

print('rmse', sqrt(mean_squared_error(y_test, y_pred)))
print('r2 score', r2_score(y_test, y_pred))
#### r2 score 0.228655730110208
#### rmse 170.01262252161752


########## Ada boost Regressor

ada_reg=AdaBoostRegressor()

param_grid2=[{'n_estimators':[22,40,55,66,90], 'learning_rate':[0.2,0.5,0.8,1.5], 'loss':['linear','square']}]

grid_search2=GridSearchCV(ada_reg, param_grid2, cv=5, verbose=3, n_jobs=4)


grid_search2.fit(X_train,y_train)


grid_search2.best_params_

ada_reg2=AdaBoostRegressor(n_estimators=22, loss='linear', learning_rate=0.5)
ada_reg2.fit(X_train, y_train)
y_pred2=ada_reg2.predict(X_test)

print('rmse', sqrt(mean_squared_error(y_test, y_pred2)))
print('r2 score', r2_score(y_test, y_pred2))
##### r2 score 0.2068529
#### rmse 172.39866

###### Bagging Regressor

bag_reg=BaggingRegressor()

param_grid3=[{'n_estimators':[5,10,18,25,35,42,51,59,67,80], 'max_samples': [2,3,4,5,7,10], 'max_features': [2,3,4,6,8,10], 'n_jobs': [2,3,4,5,6], 'verbose': [1,2,3,4,5]}]

grid_search3=GridSearchCV(bag_reg, param_grid3, cv=5, verbose=3, n_jobs=3)

grid_search3.fit(X_train, y_train)

grid_search3.best_params_

bag_reg2=BaggingRegressor(n_estimators=25, max_features=3, max_samples=3, verbose=3, n_jobs=2 )

bag_reg2.fit(X_train, y_train)
y_pred3=bag_reg2.predict(X_test)

print('rmse',sqrt(mean_squared_error(y_test, y_pred3)))
print('r2 score', r2_score(y_test, y_pred3))

#### R2 score r2 score 0.01787
#### RMSE 191.8407


