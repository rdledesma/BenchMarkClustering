#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:51:04 2023

@author: dario
"""
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

d = pd.read_csv('sa_15_vimod.csv')

d = d[d.ghi>5]
import matplotlib.pyplot as plt

plt.plot(d.ghi, d.GHI, '.r')



for c in d.cls.unique():
    model = smf.glm(formula = "kc ~ Mak + alpha + kcmod + ktmod + ktmod:kcmod + ktmod:alpha + Mak:alpha + kcmod:alpha + ktmod:Mak + Mak:kcmod", 
                    data = d[(d.Train) & (d.cls == c)], 
                    family = sm.families.Binomial())
    
    # Fit the model
    
    result = model.fit()
    
    # Display and interpret results
    # print(result.summary())
    predictions = result.predict(d[(~ d.Train) & (d.cls == c)][[ 'Mak', 'alpha',  'kcmod', 'ktmod', 'ktmod:kcmod', 'ktmod:alpha',
           'Mak:alpha', 'kcmod:alpha', 'ktmod:Mak', 'Mak:kcmod']])
    
    
     
    d[f'ghiPred_{c}'] = predictions * d[~d.Train]['Clear sky GHI']
    
    #plt.plot(d.ghi, d.ghiPred, '.r')

import numpy as np
d['ghiPredDiego'] = np.nan
for c in d.cls.unique():    
    d['ghiPredDiego'] = np.where(d.cls == c, d[f'ghiPred_{c}'], d.ghiPredDiego)

d['date'] = pd.to_datetime(d.date)
plt.plot(d.date, d.ghi, '-b')
plt.plot(d.date, d.ghiPredDiego, '-r')
plt.plot(d.date, d.GHI, '-g')



plt.plot(d.ghi, d.ghiPredDiego, '.r')
    
    
    

from scipy import interpolate
def ecdf(x): # empirical CDF computation
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

def QuantileMappinBR(y_obs,y_mod): # Bias Removal using empirical quantile mapping
    y_cor = y_mod
    x_obs,cdf_obs = ecdf(y_obs)
    x_mod,cdf_mod = ecdf(y_mod)
    # Translate data to the quantile domain, apply the CDF operator
    cdf = interpolate.interp1d(x_mod,cdf_mod,kind='nearest',fill_value='extrapolate')
    qtile = cdf(y_mod)
    # Apply de CDF^-1 operator to reverse the operation to radiation domain
    cdfinv = interpolate.interp1d(cdf_obs,x_obs,kind='nearest',fill_value='extrapolate')
    y_cor = cdfinv(qtile)
    return y_cor


for c in d.cls.unique():
    d[f'ghi_adapted{c}'] = QuantileMappinBR(d[d.cls == c].ghi.values, d.ghiPredDiego.values)


d['ghi_adapted'] = np.nan
for c in d.cls.unique():    
    d['ghi_adapted'] = np.where(d.cls == c, d[f'ghi_adapted{c}'], d.ghi_adapted)


d['date'] = pd.to_datetime(d.date)
plt.plot(d.date, d.ghi, '-b')
plt.plot(d.ghi, QuantileMappinBR(d.ghi, d.ghiPredDiego), '.r')
plt.plot(d.date, d.GHI, '-g')




import numpy as np
def mbe(true, pred):
    mbe_loss = np.sum(pred - true)/true.size
    return mbe_loss

def rmsd(true, pred):
    return np.sqrt(sum((pred - true) ** 2) / true.size)


def rmbe(true, pred):
    mbe_loss = np.sum(pred - true)/true.size
    return mbe_loss/ true.mean() * 100

def rrmsd(true, pred):
    return np.sqrt(sum((pred - true) ** 2) / true.size)/true.mean()  * 100


df = d[['ghi', 'ghiPredDiego','GHI']].dropna()

rmbe(df.ghi, df.ghiPredDiego)
rrmsd(df.ghi,df.ghiPredDiego)

