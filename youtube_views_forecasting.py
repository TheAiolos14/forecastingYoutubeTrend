#!/usr/bin/env python
# coding: utf-8

# In[46]:


import sys 
import pickle
from datetime import datetime, timedelta
from itertools import product

import sklearn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from pandas.plotting import register_matplotlib_converters
import pmdarima as pm

register_matplotlib_converters()

import math
from math import sqrt 
from numpy import inf

import warnings
warnings.filterwarnings("ignore")

import statistics 
import random

from datetime import datetime, timedelta

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from math import sqrt 
from sklearn.metrics import mean_squared_error


# In[12]:


dataset = pd.read_csv("CAvideos.csv")


# In[14]:


dataset.head()


# In[34]:


source = dataset
source = source.sort_values('publish_time')
df_target = source.groupby(["channel_title", "publish_time"])['views'].sum().to_frame().reset_index()
min_date = str(source.publish_time.min())[:10]
df_target['week'] = source.publish_time.apply(lambda x: (datetime.strptime(str(x)[0:10], '%Y-%m-%d') - datetime.strptime(min_date, "%Y-%m-%d")).days // 7)
df_target = df_target.drop(['publish_time'], axis = 1)
df_target = df_target.groupby(["channel_title", "week"])['views'].sum().to_frame().reset_index()


# In[35]:


df_target.head()


# In[42]:


list_channel_title = list(df_target.channel_title.unique())

len(list_channel_title)


# In[38]:


df_target.week.max()


# In[113]:


get_ipython().run_cell_magic('time', '', "\ny_true_final_arima = []\ny_pred_final_arima = []\nz = 1\n\nx=0\nwindow = 15\nstart_week = 165\nend_week = 530\ntarget_week = 531\nx_y = 0\n\nfor i in list_channel_title:\n    df = df_target.loc[df_target.channel_title == i] \n    x_y = 0\n\n    for x in range(1):\n        df = df.loc[(df.week >= start_week) & (df.week <= end_week)] \n        df_true = df.loc[(df.week == target_week)]            \n        j = 0\n        y_true_sampling = []\n        list_x = []\n            \n        data_test = df.drop(['channel_title'], axis= 1)\n        data_test.index = data_test['week']\n        test_array = data_test.to_numpy()\n            \n        from sklearn.preprocessing import StandardScaler\n        scaler = StandardScaler()\n            \n        df_x = list(data_test.views.values)\n        \n        if(len(df_x) < 1):\n            df_x = [10] * 365\n        else:\n            df_x = df_x\n               \n\n        sklearn = df_x\n        reshape = np.reshape(sklearn, (-1,1))\n        scaler.fit(reshape)\n            \n        reshape_final = scaler.transform(reshape)\n        reshape_final_2 = np.reshape(reshape_final, (-1))\n\n        lists_final_2 = list(reshape_final_2)\n        min_list_value = min(lists_final_2)\n\n        for t in range(1, 366):\n            if(lists_final_2.index == t):\n                list_x.append(lists_final_2[t])\n            else: \n                list_x.append(min_list_value)\n\n        model = pm.auto_arima(list_x, start_p=0, start_q=0,\n                  test='adf',   \n                  max_p=1, max_q=1, \n                  m=1,  \n                  d=1,   \n                  seasonal=False,   \n                  start_P=1, \n                  D=1, \n                  trace=False,\n                  error_action='ignore',  \n                  suppress_warnings=True, \n                  stepwise=True)\n        n_periods = 1\n\n        fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)\n        index_of_fc = np.arange(len(lists_final_2), len(lists_final_2)+n_periods)\n        fc_series = pd.Series(fc, index=index_of_fc)\n\n        fc_series_inverse = scaler.inverse_transform(fc_series.values)\n        list_y_pred = list(np.round(fc_series_inverse, 0))\n\n        list_y_pred = [0 if i < 0 else i for i in list_y_pred]\n\n        y_true = list(df_true.views.values)\n\n        if(1 > len(y_true)):\n            y_true_sampling = y_true\n            for j in range(0, (1-len(y_true))):\n                y_true_sampling.append(0)\n        else:\n            for j in range(1):\n                y_true_sampling.append(y_true[j])\n\n        y_true_final_arima += y_true_sampling\n        y_pred_final_arima += list_y_pred\n\n        start_week += 1\n        end_week += 1\n        target_week += 1\n        x_y += 1\n        \n    start_week = 165\n    end_week = 530\n    target_week = 531    \n        \n    print(z, i)\n    z += 1")


# In[131]:


x_pred_arima = y_pred_final_arima
pred_without_nan_arima = [0 if math.isnan(x) else x for x in x_pred_arima]


# In[132]:


import matplotlib.pyplot as plt

plt.plot(pred_without_nan_arima, color="red", label="prediction")
plt.legend()
plt.show()


# In[142]:


df_true = df_target.loc[(df_target.week == 531)] 


# In[143]:


pred_without_nan_arima[5071]


# In[149]:


df_final = pd.DataFrame([['종합뉴스	', 120682, 75953]], columns=["channel_title", "views_true","views_pred"])


# In[150]:


df_final


# Berdasarkan hasil analisa diatas bahwa ARIMA mampu melakukan prediksi dengan cukup baik, hal ini menandakan bahwa "종합뉴스" memiliki views yang lumayan banyak sehingga dapat membantu channel youtube terkait untuk menentukan isi konten maupun durasi sehingga dapat meningkatkan views di youtube channel terkait

# # fin
