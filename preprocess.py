# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 00:26:45 2021

@author: ASUS
"""
import pandas as pd
import numpy as np

anpr_df = pd.read_csv(
                      'D:/University related/Memarnezhad/OD_prediction/data'\
                      '/final_data/ANPR_data/mir98s3/ANPR_data_93_new_sum.csv'
                      
                      )

# Jalali to gregorian. this way calculation is way easier    
def jalali_to_gregorian(jy, jm, jd):
 jy += 1595
 days = -355668 + (365 * jy) + ((jy // 33) * 8) + (((jy % 33) + 3) // 4) + jd
 if (jm < 7):
  days += (jm - 1) * 31
 else:
  days += ((jm - 7) * 30) + 186
 gy = 400 * (days // 146097)
 days %= 146097
 if (days > 36524):
  days -= 1
  gy += 100 * (days // 36524)
  days %= 36524
  if (days >= 365):
   days += 1
 gy += 4 * (days // 1461)
 days %= 1461
 if (days > 365):
  gy += ((days - 1) // 365)
  days = (days - 1) % 365
 gd = days + 1
 if ((gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0)):
  kab = 29
 else:
  kab = 28
 sal_a = [0, 31, kab, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
 gm = 0
 while (gm < 13 and gd > sal_a[gm]):
  gd -= sal_a[gm]
  gm += 1
  final_date = '-'.join([str(gy), str(gm), str(gd)])
 return final_date


def to_gregorian(x):
    list_date = x.split('/')
    year = int(list_date[0])
    month = int(list_date[1])
    day = int(list_date[2])
    new_date = jalali_to_gregorian(year, month, day)
    return new_date

# Change all dates to gregorian
anpr_df['date']= anpr_df.apply(lambda row: to_gregorian(row['PERSIAN_DATE_STRING']), axis=1)

# We don't need this column anymore
del anpr_df['PERSIAN_DATE_STRING']

# Changing column name to be consistent with other simmilar codes I write in this project
anpr_df.rename(columns={
                        'HOUR24':'time', 'CNT':'count',
                        'traffic_zone_93': 'zone_93'
                        }, inplace=True)

def date_time_process(df):
    """what this function does:
        1- convert to pandas format datetime
        2- calculate day of the week
        return the processed dataframe"""   
    # Convert to pandas date format
    df['date'] = pd.to_datetime(df['date'])
    # Get week day
    df['week_day']= df['date'].dt.day_name()
    return df

def keep_mid_week(df):
    """get a dataframe with only Sunday, Monday
    and Tuesday"""
    df = df.loc[df.week_day.isin(['Sunday', 'Monday', 'Tuesday'])]
    return df

# 
anpr_df = date_time_process(anpr_df)

# Remove unwanted weekdays
anpr_df = keep_mid_week(anpr_df)


###############################################################################

## Remove unwanted days

# Havadese Aban 98    
start_remove = pd.to_datetime('2019-11-16')
end_remove = pd.to_datetime('2019-11-22')

anpr_df = anpr_df.query('date < @start_remove or date > @end_remove')

# Holidays
start_remove_2 = pd.to_datetime('2019-10-27')
end_remove_2 = pd.to_datetime('2019-10-29')

anpr_df = anpr_df.query('date < @start_remove_2 or date > @end_remove_2')

###############################################################################

## Calculate mean value for each taz and hour combination

anpr_df = anpr_df.groupby(['zone_93', 'time'])['count'].mean().reset_index()

anpr_df.zone_93.replace(0, np.nan, inplace=True)
anpr_df.dropna(inplace=True)

anpr_df.to_csv('D:/University related/Memarnezhad/OD_prediction/code/anpr/anpr.csv')

anpr_sum = anpr_df.groupby(['time'] )['count'].sum()
anpr_sum = pd.DataFrame({'time': anpr_sum.index, 'sum_count_anpr':anpr_sum.values})

anpr_sum.to_csv('D:/University related/Memarnezhad/OD_prediction/code/Models/anpr_sum_systemkey.csv' ,index=False)


"""
AVL data prepration

*AVL: entry and exit counts for Tehran subway stations aggregated as
 TAZ counts
 
 we consider only month in fall 
 
"""

# Put all dataframes in 2 lists, one for entry and one for exit
entry_df = pd.DataFrame()
exit_df = pd.DataFrame()

for i in range (1, 4):  
    i = str(i) # Change to string so that can be concatenated
    # Import the TAZ stations' entry count
    entry_df_x = pd.read_csv(
        'D:/University related/Memarnezhad/OD_prediction' \
        '/data/final_data/AVL_data/98fal'+ i +'/zone_93_entry.csv',
        index_col=0
        )
    if len(entry_df) == 0:
        entry_df = entry_df_x
    else:
        entry_df = pd.concat([entry_df, entry_df_x], axis =0)
            
    # Import the TAZ stations' exit count
    exit_df_x = pd.read_csv(
        'D:/University related/Memarnezhad/OD_prediction' \
        '/data/final_data/AVL_data/98fal'+ i +'/zone_93_exit.csv',
        index_col=0
        )
        
    if len(exit_df) == 0:
        exit_df = exit_df_x
    else:
        exit_df = pd.concat([exit_df, entry_df_x], axis =0)
# 0 for count means broken device at the intersection so they need to go
entry_df['count'].replace(0, np.nan, inplace=True)
exit_df['count'].replace(0, np.nan, inplace=True)
entry_df.dropna(inplace=True)    
exit_df.dropna(inplace=True) 
def date_time_process(df):
    """what this function does:
        1- split date and time 
        2- convert to pandas format datetime
        3- calculate day of the week
        4- change hours to integer numbers
        return the processed dataframe"""   
    # Split date and time
    df[['date', 'time']] = df['time'].str.split(' ', 1, expand=True)
    # Convert to pandas date format
    df['date'] = pd.to_datetime(df['date'])
    # Get week day
    df['week_day']= df['date'].dt.day_name()
    # replace an integer number instead of hour
    df['time'] = df['time'].apply(lambda row: int(row.split(':')[0]))
    return df

def keep_mid_week(df):
    """get a dataframe with only Sunday, Monday
    and Tuesday"""
    df = df.loc[df.week_day.isin(['Sunday', 'Monday', 'Tuesday'])]
    return df

# 
entry_df = date_time_process(entry_df)
exit_df = date_time_process(exit_df)

# Remove unwanted weekdays
entry_df = keep_mid_week(entry_df)
exit_df = keep_mid_week(exit_df)

###############################################################################

## Remove unwanted days

# Havadese Aban 98    
start_remove = pd.to_datetime('2019-11-16')
end_remove = pd.to_datetime('2019-11-22')

entry_df = entry_df.query('date < @start_remove or date > @end_remove')
exit_df = exit_df.query('date < @start_remove or date > @end_remove')

# Holidays
start_remove_2 = pd.to_datetime('2019-10-27')
end_remove_2 = pd.to_datetime('2019-10-29')

entry_df = entry_df.query('date < @start_remove_2 or date > @end_remove_2')
exit_df = exit_df.query('date < @start_remove_2 or date > @end_remove_2')

###############################################################################

## Calculate mean value for each taz and hour combination

entry_df = entry_df.groupby(['zone_93', 'time'])['count'].mean().reset_index()
exit_df = exit_df.groupby(['zone_93', 'time'])['count'].mean().reset_index()

entry_df.to_csv('D:/University related/Memarnezhad/OD_prediction/code/avl/avl_entry.csv')
exit_df.to_csv('D:/University related/Memarnezhad/OD_prediction/code/avl/avl_exit.csv')

avl_en_sum = entry_df.groupby(['time'] )['count'].sum()
avl_en_sum = pd.DataFrame({'time': avl_en_sum.index, 'sum_count_avl_en':avl_en_sum.values})

avl_ex_sum = exit_df.groupby(['time'] )['count'].sum()
avl_ex_sum = pd.DataFrame({'time': avl_ex_sum.index, 'sum_count_avl_ex':avl_ex_sum.values})


from datetime import datetime


# Name of all dates as list
showlist = [
             '2019-10-12.csv', '2019-10-11.csv', '2019-10-05.csv', '2019-10-15.csv',
             '2019-10-19.csv', '2019-10-14.csv', '2019-11-13.csv', '2019-11-10.csv',
             '2019-10-08.csv', '2019-11-15.csv', '2019-10-09.csv', '2019-10-21.csv',
             '2019-10-23.csv', '2019-10-07.csv', '2019-10-16.csv', '2019-11-09.csv',
             '2019-10-10.csv', '2019-10-20.csv', '2019-10-24.csv', '2019-10-18.csv',
             '2019-10-22.csv', '2019-10-06.csv', '2019-11-12.csv', '2019-10-25.csv',
             '2019-11-14.csv', '2019-10-13.csv', '2019-11-11.csv', '2019-10-17.csv'
             ]

# Get day of the week based on date
def cal_day(showlist=showlist):
  n = 0
  week_dict= dict()
  for i in showlist:
    n += 1
    year = int(i[0:4])
    month = int(i[5:7])
    day = int(i[8:10])
    wk_day = datetime(year, month, day).strftime('%A')
    if wk_day not in week_dict.keys():
      week_dict[(wk_day)] = [i]
    else:
      week_dict[(wk_day)].append(i)
  return week_dict

# Run the function      
week_day_dict = cal_day()

# Only keep the days we want
filtered_dict = {key: week_day_dict[key] for key in week_day_dict.keys()
                 &{'Sunday', 'Monday', 'Tuesday'}}

read_list = filtered_dict['Sunday'] + filtered_dict['Monday'] + \
            filtered_dict['Tuesday']

# Make an empty dataframe and add all data to it            
n_df = pd.DataFrame()
for i in read_list:  
    df = pd.read_csv(
        'D:/University related/Memarnezhad/OD_prediction' \
        '/data/final_data/Neshan/Final_OD_Hourly_matrix/'+ i,
        index_col=0
        )
    if len(n_df) == 0:
        n_df = df
    else:
        n_df = pd.concat([n_df, df], axis =0)
n_df.reset_index(drop=True, inplace=True)

n_df.rename(columns={'hour':'time', 'trip_count':'count'}, inplace=True)           

n_df = n_df.groupby(['Origin',  'Destination',  'time'])['count'].agg(['mean'])
n_df = n_df.rename(columns={'mean':'count'}).reset_index()

n_df.to_csv('D:/University related/Memarnezhad/OD_prediction/code/neshan/neshan.csv')


# Put all dataframes in 2 lists, one for entry and one for exit
scat_df = pd.DataFrame()

cols = ['date', 'zone_93', 'ave_count_93']
for i in range (1, 23):
    i = str(i) # Change to string so that can be concatenated
    # Import the TAZ stations' entry count
    df_x = pd.read_csv(
        'D:/University related/Memarnezhad/OD_prediction' \
        '/data/final_data/scats_data/scats_data/zone '+i+'/zone_93_scats_df.csv',
        usecols=cols, index_col=0
        )
    df_x.reset_index(inplace=True)
    if len(scat_df) == 0:
        scat_df = df_x
    else:
        scat_df = pd.concat([scat_df, df_x], axis =0)
        
# 0 for count means broken device at the intersection so they need to go
scat_df['ave_count_93'].replace(0, np.nan, inplace=True)
scat_df.dropna(inplace=True)

def date_time_process(df):
    """what this function does:
        1- split date and time 
        2- convert to pandas format datetime
        3- calculate day of the week
        4- change hours to integer numbers
        return the processed dataframe"""   
    # Split date and time
    df[['date', 'time']] = df['date'].str.split(' ', 1, expand=True)
    # Convert to pandas date format
    df['date'] = pd.to_datetime(df['date'])
    # Get week day
    df['week_day']= df['date'].dt.day_name()
    # replace an integer number instead of hour
    df['time'] = df['time'].apply(lambda row: int(row.split(':')[0]))
    return df

def keep_mid_week(df):
    """get a dataframe with only Sunday, Monday
    and Tuesday"""
    df = df.loc[df.week_day.isin(['Sunday', 'Monday', 'Tuesday'])]
    return df 


scat_df = date_time_process(scat_df)

# Remove unwanted weekdays
scat_df = keep_mid_week(scat_df)

###############################################################################

## Remove unwanted days

# Havadese Aban 98    
start_remove = pd.to_datetime('2019-11-16')
end_remove = pd.to_datetime('2019-11-22')

scat_df = scat_df.query('date < @start_remove or date > @end_remove')

# Holidays
start_remove_2 = pd.to_datetime('2019-10-27')
end_remove_2 = pd.to_datetime('2019-10-29')

scat_df = scat_df.query('date < @start_remove_2 or date > @end_remove_2')
scat_df.reset_index(inplace=True, drop=True)
###############################################################################

# Change intersection-stacked zones with individual zone count
scat_df['zone_93'] = scat_df['zone_93'].map(lambda x: x.lstrip('[').rstrip(']'))
expanded_zones = scat_df['zone_93'].str.split(' ', expand=True)
expanded_zones.fillna(0, inplace=True)
scat_df = pd.concat([scat_df, expanded_zones], axis=1)

# Make an empty dataframe make slices and concat them with
#their time and count. append each concated dataframe to 
#the empty dataframe
scat_df_new = pd.DataFrame(columns=['zone_93','time','count'])

for i in scat_df.iloc[:,5:].columns:
    temp_df = scat_df[[i, 'time', 'ave_count_93']]
    temp_df.rename(columns={i:'zone_93', 'ave_count_93':'count'},
                   inplace=True)
    if len(scat_df_new) == 0:
        scat_df_new = temp_df
    else:
        scat_df_new = pd.DataFrame(pd.concat([scat_df_new, temp_df],
             axis=0), columns=['zone_93','time','count'])

# Drop 0. they are non existing zones
scat_df_new.zone_93.replace(0, np.nan, inplace=True)  
scat_df_new.dropna(inplace=True)
scat_df_new.reset_index(drop=True, inplace=True)

# Calculate the mean for zone and time combinations
scat_df_new = scat_df_new.groupby(['zone_93',  'time'])['count'].agg(['mean'])
scat_df_new = scat_df_new.rename(columns={'mean':'count'}).reset_index()
# Removing empty zone code!? :/
scat_df_new.replace('', np.nan, inplace=True)
scat_df_new.dropna(inplace=True)

#scat_df_new.to_csv('D:/University related/Memarnezhad/OD_prediction/code/scats/scats.csv')


scat_sum = scat_df_new.groupby(['time'] )['count'].sum()
scat_sum = pd.DataFrame({'time': scat_sum.index, 'sum_count_scat':scat_sum.values})

pb_df = pd.read_csv('D:/University related/Memarnezhad/OD_prediction/data'\
                    '/final_data/OD_data/pb_df.csv')
pv_df = pd.read_csv('D:/University related/Memarnezhad/OD_prediction/data'\
                    '/final_data/OD_data/pv_df.csv')

pb_df = pb_df[
    [
     'Origin', 'Destination', 'q_0', 'q_1', 'q_2', 'q_3', 'q_4',
     'q_5', 'q_6', 'q', 'q_8', 'q_9', 'q_10', 'q_11', 'q_12','q_13', 'q_14',
     'q_15','q_16', 'q_17', 'q_18', 'q_19', 'q_20', 'q_21','q_22', 'q_23'
     ]
    ]

pv_df = pv_df[
    [
     'Origin', 'Destination', 'q_0', 'q_1', 'q_2', 'q_3', 'q_4',
     'q_5', 'q_6', 'q', 'q_8', 'q_9', 'q_10', 'q_11', 'q_12','q_13', 'q_14',
     'q_15','q_16', 'q_17', 'q_18', 'q_19', 'q_20', 'q_21','q_22', 'q_23'
     ]
    ]

# Rename hours

# Public
pb_df.rename(columns={'q':'q_7'},inplace=True)
pb_df.rename(
    columns=lambda x: x.split('_')[1].replace('q','') \
    if '_' in x else x, inplace=True)

# Personal
pv_df.rename(columns={'q':'q_7'},inplace=True)
pv_df.rename(
    columns=lambda x: x.split('_')[1].replace('q','') \
    if '_' in x else x, inplace=True)



def vertical_to_horizontal(df):
    df_new = pd.DataFrame(columns=['Origin','Destination', 'time', 'count'])
    for i in df.iloc[:,2:].columns:
        temp_df = df[['Origin','Destination', i]]
        temp_df['time'] = int(i)
        temp_df.rename(columns={i:'count'},
                       inplace=True)
        if len(df) == 0:
            df_new = temp_df
        else:
            df_new = pd.DataFrame(pd.concat([df_new, temp_df],
                 axis=0), columns=['Origin','Destination', 'time', 'count'])
    return df_new
            
pv_df = vertical_to_horizontal(pv_df)
pb_df = vertical_to_horizontal(pb_df)


pv_df.reset_index(drop=True, inplace=True)
pb_df.reset_index(drop=True, inplace=True)

pb_df.to_csv('D:/University related/Memarnezhad/OD_prediction/code/od_matrix/pb_df_hour.csv')
pv_df.to_csv('D:/University related/Memarnezhad/OD_prediction/code/od_matrix/pv_df_hour.csv')


from statistics import mean

# Read all necessary files
n_df = pd.read_csv('D:/University related/Memarnezhad/OD_prediction/'\
                   'code/neshan/neshan.csv', index_col=0)# Neshan
    
scat_df = pd.read_csv('D:/University related/Memarnezhad/OD_prediction'\
                      '/code/scats/scats.csv', index_col=0)
    
avl_entry_df = pd.read_csv('D:/University related/Memarnezhad/OD_prediction'\
                           '/code/avl/avl_entry.csv', index_col=0)
    
avl_exit_df = pd.read_csv('D:/University related/Memarnezhad/OD_prediction'\
                          '/code/avl/avl_exit.csv', index_col=0)
    
anpr_df = pd.read_csv('D:/University related/Memarnezhad/OD_prediction/code'\
                      '/anpr/anpr.csv', index_col=0)

# Tarh jameh (target value)  
pv_df = pd.read_csv('D:/University related/Memarnezhad/OD_prediction/code'\
                      '/od_matrix/pv_df_hour.csv', index_col=0)
pb_df = pd.read_csv('D:/University related/Memarnezhad/OD_prediction/code'\
                      '/od_matrix/pb_df_hour.csv', index_col=0)

zone_neighbors = pd.read_excel('D:/University related/Memarnezhad/'\
                               'OD_prediction/code/data aggregation/zone_neighbor.xls')

# Add zones that are not existent then put 0 az their count

# First make a list of all available TAZ's
list_navahi = list()
for i in range(1,700):
    list_navahi.append(i)

# First 24 hours  
list_hour = list()
for x in range(0,24):
    list_hour.append(x)
    

# Every combination of zone-hour combination
comb_list = list(itertools.product(list_navahi, list_hour))

def add_absent_zones(df, comb_list=comb_list, list_hour=list_hour):
    """ find and add absent zone-hour combination to the main data"""
    # Add missing hours of available zones and set it to 0
    ava_zones = list(set(df.zone_93.tolist()))
    comb_list_ava = list(itertools.product(ava_zones, list_hour))

    df_comb = df[['zone_93','time']].values 
    df_comb = [tuple(b) for b in df_comb]
    # Missing hours of current zones
    absent_combs_ava = set(comb_list_ava) - set(df_comb)
    absent_comb_df_ava = pd.DataFrame(absent_combs_ava , columns=['zone_93', 'time'])
    absent_comb_df_ava['count'] = 0
    df = pd.concat([df, absent_comb_df_ava], axis=0)
    # All missing zone-hour combinations
    df_comb = df[['zone_93','time']].values 
    df_comb = [tuple(b) for b in df_comb]
    absent_combs = set(comb_list) - set(df_comb)
    absent_comb_df = pd.DataFrame(absent_combs , columns=['zone_93', 'time'])
    absent_comb_df['count'] = -9
    df = pd.concat([df, absent_comb_df], axis=0)
    df = df[df['zone_93'] <= 699]
    df.reset_index(drop=True, inplace=True)
    return df
# Every combination of origin-destination-hour combination
comb_list_od = list(itertools.product(list_navahi,list_navahi, list_hour))

def add_absent_taz_comb(df, comb_list_od=comb_list_od):
    """ find and add absent origin-destination-hour combination
    to the main data"""
    df_comb = df[['Origin', 'Destination', 'time']].values
    df_comb = list(map(tuple, df_comb))
    absent_combs = set(comb_list_od) - set(df_comb)
    absent_comb_df = pd.DataFrame(
        absent_combs , columns=[
        'Origin', 'Destination', 'time'
        ]
        )
    absent_comb_df['count'] = 0
    df = pd.concat([df, absent_comb_df], axis=0)
    df = df[df['Origin'] <= 699]
    df = df[df['Destination'] <= 699]
    df.reset_index(drop=True, inplace=True)
    return df

# Run the above funtions on the datasets
# First func
scat_df = add_absent_zones(scat_df)
avl_entry_df = add_absent_zones(avl_entry_df)
avl_exit_df = add_absent_zones(avl_exit_df)
anpr_df = add_absent_zones(anpr_df)
# Second func
n_df = add_absent_taz_comb(n_df)
pv_df = add_absent_taz_comb(pv_df)
pb_df = add_absent_taz_comb(pb_df)

# Make a dictionary of adjecent neighbors
n_dict = {
    k: g['neighbor-zone'].tolist() for k,g in zone_neighbors.groupby('main-zone')
    }

def cal_nan_neigh_mean(df, zone_93, time, count,
                       n_dict=n_dict):
    """calculate mean of neighbor zones for absent zones"""
    if count == -9:
        n_zones = n_dict[zone_93]# Find adjecent zones of the selected zones
        avg_df = df[(df['zone_93'].isin(n_zones)) & (df['time'].isin([time]))]
        avg_df = avg_df[avg_df['count'] != -9]
        if len(avg_df)!= 0:
            # Get the adjecent zones and hours
            avg_list = avg_df['count'].to_list()
            new_count= mean(avg_list)
            return new_count
    else:
        return count

# We want to both with cal neighbor mean and without
# cal neighbor mean so I'll copy df's and calculate on them
 

# _nm = neighbor mean
scat_df_nm = scat_df.copy()
avl_entry_df_nm = avl_entry_df.copy()
avl_exit_df_nm = avl_exit_df.copy()
anpr_df_nm = anpr_df.copy()

scat_df_nm['count'] = scat_df_nm.apply(lambda row: cal_nan_neigh_mean(
    scat_df_nm, row['zone_93'] , row['time'] , row['count']), axis=1)

avl_entry_df_nm['count'] = avl_entry_df_nm.apply(lambda row: cal_nan_neigh_mean(
    avl_entry_df_nm, row['zone_93'] , row['time'] , row['count']), axis=1)

avl_exit_df_nm['count'] = avl_exit_df_nm.apply(lambda row: cal_nan_neigh_mean(
    avl_exit_df_nm, row['zone_93'] , row['time'] , row['count']), axis=1)

anpr_df_nm['count'] = anpr_df_nm.apply(lambda row: cal_nan_neigh_mean(
    anpr_df_nm, row['zone_93'] , row['time'] , row['count']), axis=1)


# TAZ features       
taz_char = pd.read_excel('D:/University related/Memarnezhad/'\
                               'OD_prediction/data/final_data/SE1393.xlsx')
# Drop unwanted columns    
taz_char.drop(['zone', 'Norm-working', 'Nor-CarOwner'], axis=1, inplace=True)
taz_char.rename(columns={'TAZ':'zone_93'}, inplace=True)
taz_char.dropna(inplace=True)

# Merge all dataframe
# Change count names for based on the dataframe
scat_df_nm.rename(columns={'count':'count_scat'}, inplace=True)
scat_df.rename(columns={'count':'count_scat'}, inplace=True)

avl_entry_df_nm.rename(columns={'count':'count_avl_en'}, inplace=True)
avl_entry_df.rename(columns={'count':'count_avl_en'}, inplace=True)

avl_exit_df_nm.rename(columns={'count':'count_avl_ex'}, inplace=True)
avl_exit_df.rename(columns={'count':'count_avl_ex'}, inplace=True)

anpr_df_nm.rename(columns={'count':'count_anpr'}, inplace=True)
anpr_df.rename(columns={'count':'count_anpr'}, inplace=True)

n_df.rename(columns={'count':'count_neshan'}, inplace=True)
pv_df.rename(columns={'count':'count_pv'}, inplace=True)
pb_df.rename(columns={'count':'count_pb'}, inplace=True)
###########################################################################
# Merge neighbor mean dataframes
zone_cnt_nm = pd.merge(
    scat_df_nm, avl_entry_df_nm, left_on=['zone_93','time'],
    right_on=['zone_93','time'], how='left'
    )
zone_cnt_nm = pd.merge(
    zone_cnt_nm, avl_exit_df_nm, left_on=['zone_93','time'],
    right_on=['zone_93','time'], how='left'
    )
zone_cnt_nm = pd.merge(
    zone_cnt_nm, anpr_df_nm, left_on=['zone_93','time'],
    right_on=['zone_93','time'], how='left'
    )
zone_cnt_nm = pd.merge(
    zone_cnt_nm, taz_char, left_on=['zone_93'], right_on=['zone_93'], how='left'
    )
###########################################################################
# Merge not neighbor meaned dataframes
zone_cnt = pd.merge(
    scat_df, avl_entry_df, left_on=['zone_93','time'],
    right_on=['zone_93','time'], how='left'
    )
zone_cnt = pd.merge(
    zone_cnt, avl_exit_df, left_on=['zone_93','time'],
    right_on=['zone_93','time'], how='left'
    )
zone_cnt = pd.merge(
    zone_cnt, anpr_df, left_on=['zone_93','time'],
    right_on=['zone_93','time'], how='left'
    )
zone_cnt = pd.merge(
    zone_cnt, taz_char, left_on=['zone_93'], right_on=['zone_93'], how='left'
    )
##########################################################################
# Merge pv nd pb data
pv_pb_df = pd.merge(
    pv_df, pb_df, left_on=['Origin', 'Destination', 'time'],
    right_on=['Origin', 'Destination', 'time'], how='left'
    )
# sum pv and pb counts 
pv_pb_df['count_pv_pb'] = pv_pb_df['count_pv'] + pv_pb_df['count_pb']

# merge neshan data
pv_pb_df = pd.merge(
    pv_pb_df, n_df, left_on=['Origin', 'Destination', 'time'],
    right_on=['Origin', 'Destination', 'time'], how='left'
    )
###########################################################################
# Add origin zones feature 
zone_cnt_nm_o = zone_cnt_nm.add_suffix('_o')

pv_pb_df_mn = pd.merge(
    pv_pb_df, zone_cnt_nm_o, left_on=['Origin','time'],
    right_on=['zone_93_o', 'time_o'], how='left'
    )
# Add Destination zones feature 
zone_cnt_nm_d = zone_cnt_nm.add_suffix('_d')

pv_pb_df_mn = pd.merge(
    pv_pb_df_mn, zone_cnt_nm_d, left_on=['Destination','time'],
    right_on=['zone_93_d', 'time_d'], how='left'
    )

# These columns are duplicates
pv_pb_df_mn.drop([
    'zone_93_o', 'zone_93_d', 'time_o', 'time_d'
    ], axis=1, inplace=True)
########################################################################
# Add origin zones feature 
zone_cnt_o = zone_cnt.add_suffix('_o')

pv_pb_df = pd.merge(
    pv_pb_df, zone_cnt_o, left_on=['Origin','time'],
    right_on=['zone_93_o', 'time_o'], how='left'
    )
# Add destination zones feature 
zone_cnt_d = zone_cnt.add_suffix('_d')

pv_pb_df = pd.merge(
    pv_pb_df, zone_cnt_d, left_on=['Destination','time'],
    right_on=['zone_93_d', 'time_d'], how='left'
    )
# These columns are duplicates
pv_pb_df.drop([
    'zone_93_o', 'zone_93_d', 'time_o', 'time_d'
    ], axis=1, inplace=True)

# Save the output
pv_pb_df_mn.to_csv(
    'D:/University related/Memarnezhad/OD_prediction/code/data aggregation/pv_pb_df_mn.csv',
    index=False)
pv_pb_df.to_csv(
    'D:/University related/Memarnezhad/OD_prediction/code/data aggregation/pv_pb_df.csv',
    index=False)









  
    
    
























































