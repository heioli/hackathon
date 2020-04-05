#%%
from parameters import DiseaseParams, SimOpts, PlotOpts, Country_Info
from SEIR_Sim import Store_Results, SEIR_Model

#%%
import numpy as np
from glob import glob
import sys
import os
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd

path = '/home/oliver/Desktop/Covid19_Simulation/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series'

#%%
df_confirmed = pd.read_csv(path+'/time_series_covid19_confirmed_global.csv')
df_deaths = pd.read_csv(path+'/time_series_covid19_deaths_global.csv')
df_recovered = pd.read_csv(path+'/time_series_covid19_recovered_global.csv')

#%%
df_confirmed.drop(columns=['Province/State','Lat','Long'],inplace=True)
df_deaths.drop(columns=['Province/State','Lat','Long'],inplace=True)
df_recovered.drop(columns=['Province/State','Lat','Long'],inplace=True)
# %%
df_confirmed_T = df_confirmed.T[1:]
df_confirmed_T.columns = df_confirmed.T.iloc[0]
df_confirmed_T.reset_index(inplace=True)
pd.to_datetime(df_confirmed_T['index'])

df_deaths_T = df_deaths.T[1:]
df_deaths_T.columns = df_deaths.T.iloc[0]
df_deaths_T.reset_index(inplace=True)
pd.to_datetime(df_deaths_T['index'])

df_recovered_T = df_recovered.T[1:]
df_recovered_T.columns = df_recovered.T.iloc[0]
df_recovered_T.reset_index(inplace=True)
pd.to_datetime(df_recovered_T['index'])


# %%
#Plot to test how data looks like
plt.plot(pd.to_datetime(df_confirmed_T['index']),df_confirmed_T['Switzerland'])
plt.plot(pd.to_datetime(df_deaths_T['index']),df_deaths_T['Switzerland'])
plt.plot(pd.to_datetime(df_recovered_T['index']),df_recovered_T['Switzerland'])
plt.show()

#%%
df_final = df_confirmed_T[['index','Switzerland']].rename(columns={'Switzerland': "confirmed"})\
            .merge(df_deaths_T[['index','Switzerland']].rename(columns={'Switzerland': "death"}),on='index')\
            .merge(df_recovered_T[['index','Switzerland']].rename(columns={'Switzerland': "recovered"}),on='index')

#%%
#Reduce dataframe to only data where infected people are found in the data
df_final_red = df_final[df_final['confirmed']>0]


# %%
# Initialize Parameters for model
disease_parameters = DiseaseParams()
simulation_paramters = SimOpts()
country_parameters = Country_Info()

model = SEIR_Model(country_parameters.country_name, country_parameters.country_population)
results = model.run_seir(disease_parameters, simulation_paramters)

# %%
