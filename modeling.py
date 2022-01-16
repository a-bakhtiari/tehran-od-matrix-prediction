import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D, BatchNormalization
import lightgbm as lgb
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score , mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor



pv_pb_df_mn = pd.read_csv('pv_pb_df.csv'
    )

pv_pb_df_mn.replace(-9, np.nan, inplace=True)
pv_pb_df_mn = pv_pb_df_mn.drop(['Origin', 'Destination', 'count_pv', 'count_pb'], axis=1)

for i in pv_pb_df_mn.columns:
    if pv_pb_df_mn[i].isna().any() == True:
        for x in range(0,24):
            mini_df = pv_pb_df_mn[pv_pb_df_mn['time']==x]
            time_series = mini_df[i]
            pv_pb_df_mn.loc[(pv_pb_df_mn['time'] == x) & (pv_pb_df_mn[i].isnull()), i] = time_series.mean()

pv_pb_df_mn = pv_pb_df_mn[pv_pb_df_mn['count_pv_pb'] < 1100]

pv_pb_df_mn['binary'] = np.where((pv_pb_df_mn['count_pv_pb'] == 0) , 0,1)

y = pv_pb_df_mn['binary']
y = y.values

X = pv_pb_df_mn.drop(['count_pv_pb', 'binary'], axis=1)

# normalizing data
scaler_x = MinMaxScaler()
scaler_x.fit(X)
X=scaler_x.transform(X)

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.25, random_state=42)

model = Sequential()
model.add(Dense(1000, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['acc'])
print(model.summary())

history = model.fit(X_train, y_train , batch_size=1024, epochs=30, validation_data=(X_test, y_test))

y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Accuracy:",accuracy_score(y_test ,y_pred))

model.save('nn_binary_model.h5')


y = pv_pb_df_mn['count_pv_pb']
y = (y.values).reshape(-1, 1)

X = pv_pb_df_mn.drop(['count_pv_pb'], axis=1)

X.columns
# reshape the data for cnn
# make 2 duplicates for "time" and "neshan"
X['time_d'] = X['time'] 
X['count_neshan_d'] = X['count_neshan']
X['distance_d'] = X['distance']

X.rename(
    columns={
        'time':'time_o', 'count_neshan':'count_neshan_o','distance':'distance_o'
        }, inplace=True
    )

X = X[[
     'time_o', 'count_neshan_o','distance_o', 'count_scat_o',
        'count_anpr_o', 'pop_o', 'emp_pop_o',
       'karmnd_dr_mhl_shghl_o', 'veh_own_o', 'n_bussi_unit_o',
       'n_hospital_bed_o', 'n_std_in _zone_o', 'n_unistd_inzone_o', 'n_stud_o',
       'n_unistd_o', 'park_area_o', 'area_o', 'household_o',
       'office_land_use_o', 'n_office_o', 'commercial_unit_o',
       'n_commercial_o', 'medical_area_o', 'n_medical_o', 'schl_o', 'hospit_o',
       'uni_o', 'n_unistd.1_o',
       
       'time_d', 'count_neshan_d','distance_d', 'count_scat_d',
        'count_anpr_d', 'pop_d', 'emp_pop_d',
       'karmnd_dr_mhl_shghl_d', 'veh_own_d', 'n_bussi_unit_d',
       'n_hospital_bed_d', 'n_std_in _zone_d', 'n_unistd_inzone_d', 'n_stud_d',
       'n_unistd_d', 'park_area_d', 'area_d', 'household_d',
       'office_land_use_d', 'n_office_d', 'commercial_unit_d',
       'n_commercial_d', 'medical_area_d', 'n_medical_d', 'schl_d', 'hospit_d',
       'uni_d', 'n_unistd.1_d']]


# normalizing data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(X)
X=scaler_x.transform(X)
scaler_y.fit(y)
y=scaler_y.transform(y)

y = y.flatten()

X = X.reshape(X.shape[0], 2 ,28, 1)

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.25, random_state=42)

model = Sequential()
model.add(
    Conv2D(
        filters=8, kernel_size=(2,2), strides=(1,1),
        activation='relu', input_shape=(2,28,1),
        padding="same"
        )
    )
model.add(
    Conv2D(
        filters=16, kernel_size=(2,2), strides=(1,1),
        activation='relu', padding="same")
    )
model.add(MaxPool2D(pool_size=(2,2), strides=(1,1)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
print(model.summary())

np.random.seed(20)

n_epoch = 50
history = model.fit(X_train, y_train , batch_size=4096, epochs=n_epoch, validation_data=(X_test, y_test))



loss_train = history.history['mae']
loss_val = history.history['val_mae']
epochs =range(n_epoch)
plt.plot(epochs, loss_train, 'g', label='Training mae')
plt.plot(epochs, loss_val, 'b', label='validation mae')
plt.title('Training and Validation mae')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('nn_wa_plot_sex_mae.svg')
plt.show()

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs =range(n_epoch)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('nn_wa_plot_sex_loss.svg')
plt.show()


y_pred_train = model.predict(X_train)#train set
y_pred_test = model.predict(X_test)#train set

print('The r2 score of  train prediction is:', r2_score(y_train, y_pred_train))#train set
print('The r2 score of test prediction is:', r2_score(y_test, y_pred_test))

print('The rmse of  train prediction is:', mean_squared_error(y_train, y_pred_train) ** 0.5)#train set
print('The rmse of test prediction is:', mean_squared_error(y_test, y_pred_test) ** 0.5)#train set

print('The mse of  train prediction is:', mean_squared_error(y_train, y_pred_train))#train set
print('The mse of test prediction is:', mean_squared_error(y_test, y_pred_test) )#train set

print('The mae of  train prediction is:', mean_absolute_error(y_train, y_pred_train))#train set
print('The mae of test prediction is:', mean_absolute_error(y_test, y_pred_test))#train set

model.save('cnn_model.h5')


# # Lets add the zone clusters and see how the output differs
# xls = pd.ExcelFile('tabagheh-bandi-navahi-finall.xlsx')
# tabaghe_bandi_key = pd.read_excel(xls, 'Sheet1')
# shardari_key = pd.read_excel('SE1393-TAZ-Navahi-Mantaghe.xlsx')

# tabaghe_bandi_key.rename(columns={'ناحیه شهرداری': 'nahie_shahrdari', 'دسته بندی':'dastebandi'}, inplace=True)
# shardari_key.rename(columns={'ناحیه شهرداری': 'nahie_shahrdari', 'ناحیه ترافیکی':'TAZ'}, inplace=True)

# tabaghe_bandi_key = tabaghe_bandi_key[['nahie_shahrdari', 'dastebandi']]
# shardari_key = shardari_key[['nahie_shahrdari', 'TAZ']]

# # add shardari area since we can't add clusters without them
# pv_pb_df_mn = pd.merge(
#     pv_pb_df_mn, shardari_key, left_on=['Destination'],
#     right_on=['TAZ'], how='left'
#     )
# pv_pb_df_mn.rename(columns={'nahie_shahrdari': 'nahie_shahrdari_d'}, inplace=True)
# pv_pb_df_mn = pv_pb_df_mn.drop(['TAZ'], axis=1)

# pv_pb_df_mn = pd.merge(
#     pv_pb_df_mn, shardari_key, left_on=['Origin'],
#     right_on=['TAZ'], how='left'
#     )

# pv_pb_df_mn.rename(columns={'nahie_shahrdari': 'nahie_shahrdari_o'}, inplace=True)
# pv_pb_df_mn = pv_pb_df_mn.drop(['TAZ'], axis=1)

# pv_pb_df_mn.fillna(-1, inplace=True)

# # merge clusters
# pv_pb_df_mn = pd.merge(
#     pv_pb_df_mn, tabaghe_bandi_key, left_on=['nahie_shahrdari_d'],
#     right_on=['nahie_shahrdari'], how='left'
#     )
# pv_pb_df_mn.rename(columns={'dastebandi': 'dastebandi_d'}, inplace=True)
# pv_pb_df_mn = pv_pb_df_mn.drop(['nahie_shahrdari'], axis=1)

# pv_pb_df_mn = pd.merge(
#     pv_pb_df_mn, tabaghe_bandi_key, left_on=['nahie_shahrdari_o'],
#     right_on=['nahie_shahrdari'], how='left'
#     )

# pv_pb_df_mn.rename(columns={'dastebandi': 'dastebandi_o'}, inplace=True)
# pv_pb_df_mn = pv_pb_df_mn.drop(['nahie_shahrdari'], axis=1)

# pv_pb_df_mn.fillna(0, inplace=True)

# cluster_o = pd.get_dummies(pv_pb_df_mn.dastebandi_o, prefix='cluster_o_')
# cluster_d = pd.get_dummies(pv_pb_df_mn.dastebandi_d, prefix='cluster_d_')

# train_data = pd.concat(
#     [(pv_pb_df_mn.drop(['dastebandi_o','dastebandi_d'], axis=1)), cluster_o, cluster_o], axis=1)

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


d_train = lgb.Dataset(X_train, label=y_train)
d_val = lgb.Dataset(X_test, label=y_test)

# specify your configurations as a dict

params = {
    'boosting_type': 'gbdt',# in parametr mitavand ba 3 algorithm amozesh bebinad.
    #dart: coputationally expensive-
    #rf: the conventional random forest algorithm
    #gbdt: coputationally efficient- acceptable high accuracy
    'objective': 'regression', # momkene classificaion ham bashe
    'metric': {'l2', 'l1'}, # i.e accuracy 
    'learning_rate':0.06 # normal range 0.001-0.1
    # tamame parameter ha mamoolan ba tanzime parameter anjam mishe

}

clf = lgb.train(params, d_train,1000000 ,valid_sets=[d_train, d_val], early_stopping_rounds=500, verbose_eval=100)



y_pred_train = clf.predict(X_train)#train set
y_pred_test = clf.predict(X_test)#train set
print('hourly mean results:')
print('The r2 score of  train prediction is:', r2_score(y_train, y_pred_train))#train set
print('The r2 score of test prediction is:', r2_score(y_test, y_pred_test))

print('The rmse of  train prediction is:', mean_squared_error(y_train, y_pred_train) ** 0.5)#train set
print('The rmse of test prediction is:', mean_squared_error(y_test, y_pred_test) ** 0.5)#train set



# save model
joblib.dump(clf, 'lgb_hourly mean.pkl')

    

warnings.simplefilter(action='ignore', category=FutureWarning)

def plotImp(model, X , num = 24):
    feature_imp = pd.DataFrame({'Value':model.feature_importance(),'Feature':X.columns})
    plt.figure(figsize=(80, 40))
    sns.set(font_scale = 5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                        ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances-all-features.jpg')
    plt.show()

plotImp(clf, X)


data_dmatrix = xgb.DMatrix(data=X,label=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.01,
                max_depth = 5, alpha = 10, n_estimators = 10000, verbose_eval=50)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
                       
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# Import the model we are using

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 10000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)

preds = rf.predict(X_test)

r2_score(y_test, preds)

