
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

pv_pb_df_mn = pd.read_csv(
    'D:/University related/Memarnezhad/OD_prediction/code/data aggregation/pv_pb_df.csv')
pv_pb_df_mn.replace(-9, np.nan, inplace=True)
# pv_pb_df_mn.dropna(inplace=True)

pv_pb_df_mn = pv_pb_df_mn.drop(
    ['Origin', 'Destination', 'count_pv', 'count_pb'], axis=1)
pv_pb_df_mn = pv_pb_df_mn[pv_pb_df_mn['count_pv_pb'] < 1100]


for i in pv_pb_df_mn.columns:
    if pv_pb_df_mn[i].isna().any() == True:
        for x in range(0, 24):
            mini_df = pv_pb_df_mn[pv_pb_df_mn['time'] == x]
            time_series = mini_df[i]
            pv_pb_df_mn.loc[(pv_pb_df_mn['time'] == x) & (
                pv_pb_df_mn[i].isnull()), i] = time_series.mean()

df1 = df.sample(1000000)
y = df1['count_pv_pb']

X = df1.drop('count_pv_pb', axis=1)


X = X.values
imp_mean = IterativeImputer(random_state=0)
import time
start_time = time.time()
imp_mean.fit(X)
imputed_data = imp_mean.transform(X)

print("--- %s seconds ---" % (time.time() - start_time))
