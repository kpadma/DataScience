##########################################################################################
#																						 #
# INSY 5378: Data Science Project on Walmart Sales Prediction 							 #
# Coded by: 								  											 #
# Elakiya Kandasamy Chelladurai and Padmavathi Karunaiananda Sekar	 					 #
# Submitted Date: 12/05/2017					 										 #
# Dataset link: https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather/data #
#																						 #
##########################################################################################

# coding: utf-8

get_ipython().magic(u'matplotlib inline')

# Import Necessay Modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

# Read in the files
ky = pd.read_csv("key.csv")
weather = pd.read_csv("weather.csv")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Check the data 
train.shape
test.shape
weather.shape

# Data Preprocessing
# Retain only necessary columns
w = weather[['station_nbr','date','codesum','snowfall','preciptotal']]

# Check for certain codes
wn = w[w['codesum'].str.contains("RA|SN|TS|GR|PL|SH")]

# Remove the missing & trace values
wn = wn[wn['snowfall'] != 'M']
wn = wn[~(wn['snowfall'].str.contains("T"))]
wn = wn[~(wn['preciptotal'].str.contains("T"))]

# Capture rows that account for weather event
wn = wn[wn['preciptotal'].ge('1.0') | wn['snowfall'].ge('2.0')]

# Merge key & train df
df = pd.merge(train, ky, on='store_nbr')

# Merge key & test df
tdf = pd.merge(test, ky, on='store_nbr')

# Pull weather info to train
dfw = pd.merge(train, wn, on='date')

# Pull weather info to test
tdfw = pd.merge(test, wn, on='date')

# Move 'units' to the last column
cols = list(dfw)
cols.insert(7, cols.pop(cols.index('units')))
dfw = dfw.loc[:, cols]

# Remove 'codesum' column after cleaning data
dfw = dfw.drop('codesum', axis=1)
tdfw = tdfw.drop('codesum', axis=1)

# Convert all columns to numeric
dfw.snowfall = dfw.snowfall.astype(float)
tdfw.snowfall = tdfw.snowfall.astype(float)
dfw.preciptotal = dfw.preciptotal.astype(float)
tdfw.preciptotal = tdfw.preciptotal.astype(float)

# Backup the dataframe
df = dfw

# Split 'date' column to 3 fields
df['Year']=[d.split('-')[0] for d in df.date]
df['Month']=[d.split('-')[1] for d in df.date]
df['Day']=[d.split('-')[2] for d in df.date]
df.Year = df.Year.astype(int)
df.Month = df.Month.astype(int)
df.Day = df.Day.astype(int)

tdfw['Year']=[d.split('-')[0] for d in tdfw.date]
tdfw['Month']=[d.split('-')[1] for d in tdfw.date]
tdfw['Day']=[d.split('-')[2] for d in tdfw.date]
tdfw.Year = tdfw.Year.astype(int)
tdfw.Month = tdfw.Month.astype(int)
tdfw.Day = tdfw.Day.astype(int)

# drop 'date' column for prediction
df = df.drop('date', axis=1)
tdfw = tdfw.drop('date', axis=1)
cols = list(df)
cols.insert(8, cols.pop(cols.index('units')))
df = df.loc[:, cols]

# Final Dataset as csv for restoring
df.to_csv("train_final.csv", index=False)

# Final Dataset as csv for restoring -- test
tdfw.to_csv("test_final.csv", index=False)

# ### Read CSV & make DF for reuse
df = pd.read_csv("train_final.csv")
tdf = pd.read_csv("test_final.csv")

# check for correlation
corr = df.corr()

# Plot the correlation matrix
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns, annot=True, fmt=".1f")

# ### Linear Regression
x = df[['Year','Month','Day', 'store_nbr','item_nbr','station_nbr','snowfall','preciptotal']]
xt = tdf[['Year','Month','Day', 'store_nbr','item_nbr','station_nbr','snowfall','preciptotal']]
y = df['units']

lr = LinearRegression().fit(x.values,y.values)

# Get intercept and coefficient
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

print("Training set score: {:.5f}".format(lr.score(x.values, y.values)))

# Train test split
xtrain, xtest, ytrain, ytest = train_test_split(x.values,y.values, random_state=25)
lm = LinearRegression().fit(xtrain,ytrain)
print("lm.coef_: {}".format(lm.coef_))
print("lm.intercept_: {}".format(lm.intercept_))
print("Training set score: {:.5f}".format(lm.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(lm.score(xtest, ytest)))

# Train test split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, random_state=25)
rlm = Ridge(alpha=5).fit(xtrain,ytrain)

print("lm.coef_: {}".format(rlm.coef_))
print("lm.intercept_: {}".format(rlm.intercept_))
print("Training set score: {:.5f}".format(rlm.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(rlm.score(xtest, ytest)))

rlm = Ridge(alpha=10).fit(xtrain,ytrain)
print("lm.coef_: {}".format(rlm.coef_))
print("lm.intercept_: {}".format(rlm.intercept_))
print("Training set score: {:.5f}".format(rlm.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(rlm.score(xtest, ytest)))

rlm = Ridge(alpha=50).fit(xtrain,ytrain)
print("lm.coef_: {}".format(rlm.coef_))
print("lm.intercept_: {}".format(rlm.intercept_))
print("Training set score: {:.5f}".format(rlm.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(rlm.score(xtest, ytest)))

rlm = Ridge(alpha=100).fit(xtrain,ytrain)
print("lm.coef_: {}".format(rlm.coef_))
print("lm.intercept_: {}".format(rlm.intercept_))
print("Training set score: {:.5f}".format(rlm.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(rlm.score(xtest, ytest)))

rlm = Ridge(alpha=1000).fit(xtrain,ytrain)
print("lm.coef_: {}".format(rlm.coef_))
print("lm.intercept_: {}".format(rlm.intercept_))
print("Training set score: {:.5f}".format(rlm.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(rlm.score(xtest, ytest)))

# Lasso
llm = Lasso(alpha=0.1).fit(xtrain,ytrain)
print("lm.coef_: {}".format(llm.coef_))
print("lm.intercept_: {}".format(llm.intercept_))
print("Training set score: {:.5f}".format(llm.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(llm.score(xtest, ytest)))

llm = Lasso(alpha=0.05).fit(xtrain,ytrain)
print("lm.coef_: {}".format(llm.coef_))
print("lm.intercept_: {}".format(llm.intercept_))
print("Training set score: {:.5f}".format(llm.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(llm.score(xtest, ytest)))

llm = Lasso(alpha=0.005).fit(xtrain,ytrain)
print("lm.coef_: {}".format(llm.coef_))
print("lm.intercept_: {}".format(llm.intercept_))
print("Training set score: {:.5f}".format(llm.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(llm.score(xtest, ytest)))

llm = Lasso(alpha=0.0001).fit(xtrain,ytrain)
print("lm.coef_: {}".format(llm.coef_))
print("lm.intercept_: {}".format(llm.intercept_))
print("Training set score: {:.5f}".format(llm.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(llm.score(xtest, ytest)))

llm = Lasso(alpha=0.00001).fit(xtrain,ytrain)
print("lm.coef_: {}".format(llm.coef_))
print("lm.intercept_: {}".format(llm.intercept_))
print("Training set score: {:.5f}".format(llm.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(llm.score(xtest, ytest)))


# ### Decision Tree Regressor

xtrain, xtest, ytrain, ytest = train_test_split(x,y, random_state=25)

dtr = DecisionTreeRegressor(max_depth=5, random_state=10).fit(xtrain,ytrain)
print("Training set score: {:.5f}".format(dtr.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(dtr.score(xtest, ytest)))

dtr = DecisionTreeRegressor(max_depth=9, random_state=10).fit(xtrain,ytrain)
print("Training set score: {:.5f}".format(dtr.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(dtr.score(xtest, ytest)))

dtr = DecisionTreeRegressor(max_depth=10, random_state=10).fit(xtrain,ytrain)
print("Training set score: {:.5f}".format(dtr.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(dtr.score(xtest, ytest)))

dtr = DecisionTreeRegressor(max_depth=11, random_state=10).fit(xtrain,ytrain)
print("Training set score: {:.5f}".format(dtr.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(dtr.score(xtest, ytest)))

dtr = DecisionTreeRegressor(max_depth=12, random_state=10).fit(xtrain,ytrain)
print("Training set score: {:.5f}".format(dtr.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(dtr.score(xtest, ytest)))

dtr = DecisionTreeRegressor(max_depth=15, random_state=10).fit(xtrain,ytrain)
print("Training set score: {:.5f}".format(dtr.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(dtr.score(xtest, ytest)))

dtr = DecisionTreeRegressor(max_depth=18, random_state=10).fit(xtrain,ytrain)
print("Training set score: {:.5f}".format(dtr.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(dtr.score(xtest, ytest)))


# #### 2) Use chosen model on explicit train & test set
dtr = DecisionTreeRegressor(max_depth=11, random_state=10).fit(x,y)
yt = dtr.predict(xt)
print("Training set score: {:.5f}".format(dtr.score(x, y)))
print("Predictions:\n{}".format(dtr.predict(xt)))

# Print Predictions
ryt = np.around(yt)
arr = np.column_stack((xt,ryt))
pred = pd.DataFrame(arr, columns= ['Year','Month','Day','store_nbr','item_nbr','station_nbr','snowfall','preciptotal', 'units'])
pred_sorted = pred.sort_values(by='units', ascending=False)
pred_sorted.head(10)

# #### 3) Other metrics
dtr = DecisionTreeRegressor(max_depth=11, random_state=10).fit(x,y)
yo = dtr.predict(x)
print "R^2: %.3f" %r2_score(y, yo)
print "Mean Squared Error: %.3f" %mean_squared_error(y, yo)
print "Mean Absolute Error: %.3f" %mean_absolute_error(y, yo)
# if the predicted values are different, such that the mean residue IS 0: R2 = Explained variance
print "Explained Variance Score: %.3f" %explained_variance_score(y, yo)

# #### 4) CV
dtr = DecisionTreeRegressor(max_depth=11, random_state=10).fit(x,y)
crossvalidation = KFold(n=x.shape[0], n_folds=10, shuffle=True, random_state=1)
MSE = np.mean(cross_val_score(dtr, x, y,scoring='neg_mean_squared_error', cv=crossvalidation,n_jobs=1))
R2 = np.mean(cross_val_score(dtr, x, y,scoring='r2', cv=crossvalidation,n_jobs=1))
ASE = np.mean(cross_val_score(dtr, x, y,scoring='neg_mean_absolute_error', cv=crossvalidation,n_jobs=1))
print 'Mean squared error: %.3f' % abs(MSE)
print 'Mean absolute error: %.3f' % abs(ASE)
print 'R^2: %.3f' % abs(R2)


# ### Random Forest Regressor

xtrain, xtest, ytrain, ytest = train_test_split(x,y, random_state=25)
scaler = StandardScaler().fit(xtrain)
xtrain_scaled = scaler.transform(xtrain)
xtest_scaled = scaler.transform(xtest)

rfr = RandomForestRegressor(n_estimators=10).fit(xtrain_scaled,ytrain)
print("Training set score: {:.5f}".format(rfr.score(xtrain_scaled, ytrain)))
print("Test set score: {:.5f}".format(rfr.score(xtest_scaled, ytest)))

rfr = RandomForestRegressor(n_estimators=25).fit(xtrain_scaled,ytrain)
print("Training set score: {:.5f}".format(rfr.score(xtrain_scaled, ytrain)))
print("Test set score: {:.5f}".format(rfr.score(xtest_scaled, ytest)))

rfr = RandomForestRegressor(n_estimators=50).fit(xtrain,ytrain)
print("Training set score: {:.5f}".format(rfr.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(rfr.score(xtest, ytest)))

rfr = RandomForestRegressor(n_estimators=50).fit(xtrain_scaled,ytrain)
print("Training set score: {:.5f}".format(rfr.score(xtrain_scaled, ytrain)))
print("Test set score: {:.5f}".format(rfr.score(xtest_scaled, ytest)))

rfr = RandomForestRegressor(n_estimators=75).fit(xtrain_scaled,ytrain)
print("Training set score: {:.5f}".format(rfr.score(xtrain_scaled, ytrain)))
print("Test set score: {:.5f}".format(rfr.score(xtest_scaled, ytest)))

rfr = RandomForestRegressor(n_estimators=100).fit(xtrain_scaled,ytrain)
print("Training set score: {:.5f}".format(rfr.score(xtrain_scaled, ytrain)))
print("Test set score: {:.5f}".format(rfr.score(xtest_scaled, ytest)))

rfr = RandomForestRegressor(n_estimators=200).fit(xtrain_scaled,ytrain)
print("Training set score: {:.5f}".format(rfr.score(xtrain_scaled, ytrain)))
print("Test set score: {:.5f}".format(rfr.score(xtest_scaled, ytest)))

# #### 2) Use chosen model on explicit train & test set
rfr = RandomForestRegressor(n_estimators=200).fit(xtrain,ytrain)
yt = rfr.predict(xt)
print("Training set score: {:.5f}".format(rfr.score(x, y)))
print("Predictions:\n{}".format(yt))

# Print Predictions
ryt = np.around(yt)
arr = np.column_stack((xt,ryt))
pred = pd.DataFrame(arr, columns= ['Year','Month','Day','store_nbr','item_nbr','station_nbr','snowfall','preciptotal', 'units'])
pred_sorted = pred.sort_values(by='units', ascending=False)
pred_sorted.head(10)

# #### 3) Other metrics
rfr = RandomForestRegressor(n_estimators=200).fit(xtrain,ytrain)
yo = rfr.predict(x)
print "R^2: %.3f" %r2_score(y, yo)
print "Mean Squared Error: %.3f" %mean_squared_error(y, yo)
print "Mean Absolute Error: %.3f" %mean_absolute_error(y, yo)
# if the predicted values are different, such that the mean residue IS 0: R2 = Explained variance
print "Explained Variance Score: %.3f" %explained_variance_score(y, yo)

# #### 4) CV
rfr = RandomForestRegressor(n_estimators=200).fit(xtrain,ytrain)
crossvalidation = KFold(n=x.shape[0], n_folds=5, shuffle=True, random_state=1)
MSE = np.mean(cross_val_score(rfr, x, y,scoring='neg_mean_squared_error', cv=crossvalidation,n_jobs=1))
R2 = np.mean(cross_val_score(rfr, x, y,scoring='r2', cv=crossvalidation,n_jobs=1))
ASE = np.mean(cross_val_score(rfr, x, y,scoring='neg_mean_absolute_error', cv=crossvalidation,n_jobs=1))
print 'Mean squared error: %.3f' % abs(MSE)
print 'Mean absolute error: %.3f' % abs(ASE)
print 'R^2: %.3f' % abs(R2)

### AdaBoost Regressor
abr = AdaBoostRegressor(base_estimator=RandomForestRegressor(n_estimators=75), n_estimators=20,random_state=10).fit(xtrain,ytrain)
print("Training set score: {:.5f}".format(abr.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(abr.score(xtest, ytest)))

abr = AdaBoostRegressor(n_estimators=10, random_state=10).fit(xtrain,ytrain)
print("Training set score: {:.5f}".format(abr.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(abr.score(xtest, ytest)))

abr = AdaBoostRegressor(n_estimators=20, random_state=10).fit(xtrain,ytrain)
print("Training set score: {:.5f}".format(abr.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(abr.score(xtest, ytest)))

abr = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=11), n_estimators=20,random_state=10).fit(xtrain,ytrain)
print("Training set score: {:.5f}".format(abr.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(abr.score(xtest, ytest)))

abr = AdaBoostRegressor(base_estimator=RandomForestRegressor(n_estimators=50), n_estimators=20,random_state=10).fit(xtrain,ytrain)
print("Training set score: {:.5f}".format(abr.score(xtrain, ytrain)))
print("Test set score: {:.5f}".format(abr.score(xtest, ytest)))

abr = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=11), n_estimators=20,random_state=10).fit(xtrain,ytrain)
yt = abr.predict(xt)
print("Training set score: {:.5f}".format(abr.score(x, y)))
print("Predictions:\n{}".format(yt))

# Print predictions
ryt = np.around(yt)
arr = np.column_stack((xt,ryt))
pred = pd.DataFrame(arr, columns= ['Year','Month','Day','store_nbr','item_nbr','station_nbr','snowfall','preciptotal', 'units'])
pred_sorted = pred.sort_values(by='units', ascending=False)
pred_sorted.head(10)

# #### 3) Other metrics
abr = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=11), n_estimators=20,random_state=10).fit(xtrain,ytrain)
yo = abr.predict(x)
print "R^2: %.3f" %r2_score(y, yo)
print "Mean Squared Error: %.3f" %mean_squared_error(y, yo)
print "Mean Absolute Error: %.3f" %mean_absolute_error(y, yo)
# if the predicted values are different, such that the mean residue IS 0: R2 = Explained variance
print "Explained Variance Score: %.3f" %explained_variance_score(y, yo)

# #### 4) CV
abr = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=11), n_estimators=20,random_state=10).fit(xtrain,ytrain)
crossvalidation = KFold(n=x.shape[0], n_folds=10, shuffle=True, random_state=1)
MSE = np.mean(cross_val_score(abr, x, y,scoring='neg_mean_squared_error', cv=crossvalidation,n_jobs=1))
R2 = np.mean(cross_val_score(abr, x, y,scoring='r2', cv=crossvalidation,n_jobs=1))
ASE = np.mean(cross_val_score(abr, x, y,scoring='neg_mean_absolute_error', cv=crossvalidation,n_jobs=1))
print 'Mean squared error: %.3f' % abs(MSE)
print 'Mean absolute error: %.3f' % abs(ASE)
print 'R^2: %.3f' % abs(R2)
