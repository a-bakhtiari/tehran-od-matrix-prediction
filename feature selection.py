# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 23:46:44 2021


"""



# example of mutual information feature selection for numerical input data
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

pv_pb_df_mn = pd.read_csv(pv_pb_df.csv')

pv_pb_df_mn.replace(-9, np.nan, inplace=True)
pv_pb_df_mn = pv_pb_df_mn.drop(['Origin', 'Destination', 'count_pv', 'count_pb'], axis=1)

pv_pb_df_mn.dropna(inplace=True)


y = pv_pb_df_mn['count_pv_pb']
y = (y.values).reshape(-1, 1)

X = pv_pb_df_mn.drop('count_pv_pb', axis=1)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(X)
X=scaler_x.transform(X)
scaler_y.fit(y)
y=scaler_y.transform(y)

y = y.flatten()

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.25, random_state=42)

# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=mutual_info_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
n = 0
for i in X.columns:
    print(i, '----importance:', fs.scores_[n])
    n += 1
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

# example of correlation feature selection for numerical data
from sklearn.feature_selection import f_regression
 
# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=f_regression, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
 

X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
n = 0
for i in X.columns:
    print(i, '----importance:', fs.scores_[n])
    n += 1


