"""
Fit a simplified SEIR model to current Covid Data
"""

from parameters import DiseaseParams, SimOpts, PlotOpts, Country_Info
from SEIR_Sim import Store_Results, SEIR_Model

import numpy as np
from glob import glob
import sys
import os
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd

#Get most up-to-date data directly from John Hopkins University github repo
path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series'

#%%
"""
Read in data into dataframes
"""

df_confirmed = pd.read_csv(path+'/time_series_covid19_confirmed_global.csv')
df_deaths = pd.read_csv(path+'/time_series_covid19_deaths_global.csv')
df_recovered = pd.read_csv(path+'/time_series_covid19_recovered_global.csv')

df_confirmed.drop(columns=['Province/State','Lat','Long'],inplace=True)
df_deaths.drop(columns=['Province/State','Lat','Long'],inplace=True)
df_recovered.drop(columns=['Province/State','Lat','Long'],inplace=True)

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

df_final = df_confirmed_T[['index','Switzerland']].rename(columns={'Switzerland': "confirmed"})\
            .merge(df_deaths_T[['index','Switzerland']].rename(columns={'Switzerland': "death"}),on='index')\
            .merge(df_recovered_T[['index','Switzerland']].rename(columns={'Switzerland': "recovered"}),on='index')

#%%
#Reduce dataframe to only data where infected people are found in the data
df_final_red = df_final[df_final['confirmed']>0].copy()
df_final_red['current_infections'] = df_final_red['confirmed'] - df_final_red['recovered']

nr_days = len(df_final_red)

df_final_red['days_since_infection'] = np.arange(nr_days) #Get days since the first infection occurred


# %%
"""
Rewrite some of the function code used in SEIR_Sim.py to ease a simple minimization fit
"""

def model_seir(t,Y,beta,gamma,sigma,n_pop):
    """
    Simplified SEIR model
    """
   
    N = n_pop
    S = Y[0]
    E = Y[1]
    I = Y[2]
    R = Y[3]

    dS = - beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I

    return [dS, dE, dI, dR]


def run_seir(params,sim_length,n_pop):
    """
    Run ODE solver
    """

    T = np.arange(sim_length) #array with equally spaced time steps (days)
    Y0 = [n_pop - params[3], params[3], 0, 0] #S, E, I, R , params[3] is the initial exposed parameter

    Y_results = scipy.integrate.solve_ivp(model_seir, t_span=[T[0], T[-1]], 
                                            y0=Y0, model='RK45', args=(params[0],params[1],params[2],n_pop), t_eval=T)

    return Y_results


def sum_squares(params,sim_length,n_pop,initial_exposed,df_final_red):
    """
    Calculate residuals
    """
    res = run_seir(params,sim_length,n_pop)
  
    return sum((res.y[2]-df_final_red[['current_infections']].to_numpy().ravel())**2)



from scipy.optimize import minimize
#minimization step
param_init_min = [1./2.5,  1.0 / (10 + 3), 1./2.,1] #choose intial parameters
mini = minimize(sum_squares,param_init_min,method='Nelder-Mead',args=(nr_days,8.57*10**6,1,df_final_red)) #use options={'maxiter':1000} in case iterations have to be restricted

#Check
print(mini.x)
print(param_init_min)

"""
Some plotting
"""

test = run_seir(param_init_min ,nr_days,8.57*10**6)
fit_data = run_seir(mini.x,nr_days,8.57*10**6)

plt.plot(test.t,test.y[2],label='Simulation: Infected')
plt.plot(fit_data.t,fit_data.y[2],'--',label='Fitted: Infected')
plt.scatter(df_final_red[['days_since_infection']],df_final_red[['current_infections']],label='Data: Infected')
plt.scatter(df_final_red[['days_since_infection']],df_final_red[['confirmed']],label='Data: Confirmed')
plt.legend()
plt.show()


