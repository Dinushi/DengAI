#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
get_ipython().magic(u'matplotlib inline')
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,     GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from rpy2 import robjects
from rpy2.robjects import r
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# In[2]:


training_records = pd.read_csv('dengue_features_train.csv')
testing_records = pd.read_csv('dengue_features_test.csv')


# In[3]:


training_records_sj = training_records[training_records['city'] == 'sj'].drop('city', 1)
training_records_iq = training_records[training_records['city'] == 'iq'].drop('city', 1)
testing_records_sj = testing_records[testing_records['city'] == 'sj'].drop('city', 1)
testing_records_iq = testing_records[testing_records['city'] == 'iq'].drop('city', 1)
records_iq = pd.concat([training_records_iq, testing_records_iq], ignore_index=True)
records_sj = pd.concat([training_records_sj, testing_records_sj], ignore_index=True)
# Null check
#pd.isnull(training_records_iq).any()


# In[4]:


records_iq.drop(['reanalysis_avg_temp_k','reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg'], axis=1, inplace=True)
records_sj.drop(['reanalysis_avg_temp_k','reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg'], axis=1, inplace=True)
records_iq.drop(['year'], axis=1, inplace=True)
records_sj.drop(['year'], axis=1, inplace=True)


# In[5]:


records_iq.to_csv('./Files/PreProcessed-features-iq.csv', index=False)
records_sj.to_csv('./Files/PreProcessed-features-sj.csv', index=False)


# In[6]:


training_targets = pd.read_csv('dengue_labels_train.csv')


# In[7]:


training_targets.drop(['year'], axis=1, inplace=True)
training_targets['week_start_date'] = training_records['week_start_date']
training_targets_sj = training_targets[training_targets['city'] == 'sj'].drop('city', 1)
training_targets_iq = training_targets[training_targets['city'] == 'iq'].drop('city', 1)


# In[8]:


training_targets_iq.to_csv('./Files/PreProcessed-labels-train-iq.csv', index=False)
training_targets_sj.to_csv('./Files/PreProcessed-labels-train-sj.csv', index=False)


# In[9]:


records_iq = pd.read_csv(
    './Files/PreProcessed-features-iq.csv', 
    parse_dates=['week_start_date'], 
    index_col='week_start_date')
records_sj = pd.read_csv(
    './Files/PreProcessed-features-sj.csv', 
    parse_dates=['week_start_date'], 
    index_col='week_start_date')
records_iq_i1 = records_iq.interpolate()
records_sj_i1 = records_sj.interpolate()


# In[10]:


def time_series_decompose(df, column, freq=52):
    dfd = pd.DataFrame(index=df.index)
    series = robjects.IntVector(list(df[column].values))
    length = len(series)
    rts = r.ts(series, frequency=freq)
    decomposed = list(r.stl(rts, 'periodic', robust=True).rx2('time.series'))
    dfd['trend'] = decomposed[length:2*length]
    dfd['seasonal'] = decomposed[0:length]
    dfd['trend+seasonal']=pd.DataFrame(decomposed[length:2*length]).add(pd.DataFrame(decomposed[0:length]))
    dfd['residuals'] = decomposed[2*length:3*length]
    return dfd


# In[11]:


labels_iq = pd.read_csv(
    './Files/PreProcessed-labels-train-iq.csv',
    parse_dates=['week_start_date'],
    index_col='week_start_date'
)
labels_sj = pd.read_csv(
    './Files/PreProcessed-labels-train-sj.csv',
    parse_dates=['week_start_date'],
    index_col='week_start_date'
)


# In[12]:


p=labels_iq.reset_index('week_start_date')
q=time_series_decompose(p,'total_cases')
p['trend_total_cases']=q['trend']
p['seasonal_total_cases']=q['seasonal']
p['trend_seasonal_total_cases']=q['trend+seasonal']
p['residuals_total_cases']=q['residuals']
p.set_index('week_start_date',inplace=True)
p.to_csv('./Files/PreProcessed-withTrends-lables-iq.csv')


# In[13]:


p=labels_sj.reset_index('week_start_date')
q=time_series_decompose(p,'total_cases')
p['trend_total_cases']=q['trend']
p['seasonal_total_cases']=q['seasonal']
p['trend_seasonal_total_cases']=q['trend+seasonal']
p['residuals_total_cases']=q['residuals']
p.set_index('week_start_date',inplace=True)
p.to_csv('./Files/PreProcessed-withTrends-lables-sj.csv')


# In[14]:


records_iq = pd.read_csv(
    './Files/PreProcessed-features-iq.csv', 
    parse_dates=['week_start_date'],
    index_col='week_start_date'
).interpolate()
records_sj = pd.read_csv(
    './Files/PreProcessed-features-sj.csv', 
    parse_dates=['week_start_date'],
    index_col='week_start_date'
).interpolate()
labels_iq = pd.read_csv(
    './Files/PreProcessed-labels-train-iq.csv',
    parse_dates=['week_start_date'],
    index_col='week_start_date'
)
labels_sj = pd.read_csv(
    './Files/PreProcessed-labels-train-sj.csv',
    parse_dates=['week_start_date'],
    index_col='week_start_date'
)

labels_iq_timeSeries = pd.read_csv(
    './Files/PreProcessed-withTrends-lables-iq.csv',
    parse_dates=['week_start_date'],
    index_col='week_start_date'
)
labels_sj_timeSeries = pd.read_csv(
    './Files/PreProcessed-withTrends-lables-sj.csv',
    parse_dates=['week_start_date'],
    index_col='week_start_date'
)


# In[15]:


def getPredictionsTime(Id, totalRecords,labels,numOfTrain , period ,features):
    ##One hot encode weekofyear
    weeks = pd.get_dummies(totalRecords['weekofyear'], prefix='w')
    train_time , test_time = weeks[:numOfTrain].reset_index().drop('week_start_date'
                                                                 , axis=1) ,weeks[numOfTrain:].reset_index().drop('week_start_date', axis=1)
    train_cases = labels[['total_cases']].reset_index().drop('week_start_date', axis=1)
    train_cases1 = labels[['trend_total_cases']].reset_index().drop('week_start_date', axis=1)
    train_cases2 = labels[['seasonal_total_cases']].reset_index().drop('week_start_date', axis=1)
    train_cases3 = labels[['trend_seasonal_total_cases']].reset_index().drop('week_start_date', axis=1)
    train_cases4 = labels[['residuals_total_cases']].reset_index().drop('week_start_date', axis=1)

    ####Trend prediction model
    trend_model=LinearRegression()
    trend_model.fit(train_time, train_cases1)
    
    ####Seasonality prediction model
    seasonal_model = LinearRegression()
    seasonal_model.fit(train_time, train_cases2)
    
    seasonal_train = pd.Series(
        seasonal_model.predict(train_time).flatten()).rolling(5, min_periods=1, center=True).mean()
    
    trend_train=pd.Series(
        trend_model.predict(train_time).flatten()).rolling(5, min_periods=1, center=True).mean()
    
    train_residualsComponent = train_cases.total_cases- seasonal_train-trend_train
    
    residuals = totalRecords[features].reset_index().drop('week_start_date', axis=1).rolling(period).mean()
    
    train_residuals = residuals[period:numOfTrain]
    test_residuals = residuals[numOfTrain:]
    train_remainder = train_residualsComponent[period:]
    
    #Residuals prediction model
    residuals_model = LinearRegression()
    residuals_model.fit(train_residuals, train_remainder)
    train_pred_residuals = pd.Series(residuals_model.predict(train_residuals).flatten())

    print('Mean_absolute_error for example '+str(Id) +" - "+ str(mean_absolute_error(y_pred=train_pred_residuals.values + seasonal_train[period:].values+ trend_train[period:].values,
                    y_true=train_cases['total_cases'][period:].values)))
    p=pd.DataFrame()
    pred = train_pred_residuals.values + seasonal_train[period:].values +trend_train[period:].values
    real= train_cases['total_cases'][period:].values
    #p['predicted']=pred
    #p['real']=real
    #p.plot(figsize=(14, 10))

    predicted_seasonal = pd.Series(seasonal_model.predict(test_time).flatten())
    predicted_trend = pd.Series(trend_model.predict(test_time).flatten())
    predicted_residuals = pd.Series(residuals_model.predict(test_residuals).flatten())
  
    
    pred = (predicted_trend + predicted_seasonal + predicted_residuals).rolling(5, min_periods=1, center=True).mean().astype(int)
   
    p2=pd.DataFrame()
    p2['predicted'] = pred
    p2['real']=real[0:len(pred)]
    p2.plot(figsize=(14, 10))
    return pred


# In[16]:


pred_iq = getPredictionsTime(1,records_iq, labels_iq_timeSeries, 520, 53, [
    'reanalysis_precip_amt_kg_per_m2',
    'reanalysis_relative_humidity_percent', 
    'station_avg_temp_c'
])
pred_sj = getPredictionsTime(2,records_sj, labels_sj_timeSeries, 936, 53, [
    'reanalysis_precip_amt_kg_per_m2',
    'reanalysis_relative_humidity_percent',
    'station_avg_temp_c'
])


# In[17]:


pred = pd.concat([pred_sj, pred_iq], ignore_index=True).round().clip(lower=0)
pred.to_csv('./Files/Prediction-seasonal-trend-prediction-' + '.csv', index=False, header=False)


# In[ ]:




