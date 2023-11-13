#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:51:04 2023

@author: dario
"""
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

d = pd.read_csv('sa_15_Diego.csv')
d['date'] = pd.to_datetime(d.date)
c = pd.read_csv('sa_15_cony.csv')
c['date'] = pd.to_datetime(c.date)




d = (d.set_index('date')
      .reindex(c.date)
      .rename_axis(['date'])
      #.fillna(0)
      .reset_index())



d['kcmod'] = c['kcmod']
d['ktmod'] = c['ktmod']
d['ktmod:kcmod'] = c['ktmod:kcmod']
d['ktmod:alpha'] = c['ktmod:alpha']
d['Mak:alpha'] = c['Mak:alpha']
d['kcmod:alpha'] = c['kcmod:alpha']
d['ktmod:Mak'] = c['ktmod:Mak']
d['Mak:kcmod'] = c['Mak:kcmod']

d = d[d.alpha>10]
d = d[d.kc<1.3]
d = d.dropna()
d = d[d.ghi>5]
dtrain = d[d.date.dt.year == 2015]
dtest = d[d.date.dt.year < 2015]












for cls in d.clsDiego.unique():
    
    Xtrain = dtrain[dtrain.clsDiego == cls]
    
    
    
    model = smf.glm(formula = "kc ~ Mak + alpha + delta + kcmod + ktmod +  ktmod:kcmod + ktmod:alpha + Mak:alpha + kcmod:alpha + ktmod:Mak +Mak:kcmod", 
                    data = Xtrain, 
                    family = sm.families.Binomial())
    
    # Fit the model
    
    result = model.fit()
    
    # Display and interpret results
    # print(result.summary())
    predictions = result.predict(dtest) * dtest['Clear sky GHI'].values
    
     
    dtest[f'ghiPred_{cls}'] = predictions
    
    #plt.plot(d.ghi, d.ghiPred, '.r')

import numpy as np
dtest['ghiPred'] = np.nan
for c in dtest.clsDiego.unique():    
    dtest['ghiPred'] = np.where(dtest.clsDiego == c, dtest[f'ghiPred_{c}'], dtest.ghiPred)





import matplotlib.pyplot as plt

plt.plot(dtest.CTZ, dtest.ghiPred, '.r')




d['date'] = pd.to_datetime(d.date)
plt.plot(dtest.date, dtest.ghi, '-b')
plt.plot(dtest.date, dtest.ghiPred, '-r')
plt.plot(dtest.date, dtest.GHI, '-g')




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


df = dtest[['ghi', 'ghiPred']]

rmbe(df.ghi, df.ghiPred)
rrmsd(df.ghi, df.ghiPred)

df.ghi.mean()
