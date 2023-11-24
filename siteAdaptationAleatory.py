#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:12:36 2023

@author: solar
"""

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import Metrics as m
import matplotlib.pyplot as plt

import solarforecastarbiter.metrics.deterministic as det



maes = []

dferrors = pd.DataFrame()
c = pd.read_csv('sa_15_cony.csv')
c['date'] = pd.to_datetime(c.date)





dfmae = pd.DataFrame(columns=[f"met_{i}" for i in range(2,13)])
dfmbe = pd.DataFrame(columns=[f"met_{i}" for i in range(2,13)])
dfrmsd = pd.DataFrame(columns=[f"met_{i}" for i in range(2,13)])
dfksi = pd.DataFrame(columns=[f"met_{i}" for i in range(2,13)])
dfover = pd.DataFrame(columns=[f"met_{i}" for i in range(2,13)])
dfcpi = pd.DataFrame(columns=[f"met_{i}" for i in range(2,13)])
for i in range(0, 100):
        
    
    d = pd.read_csv(f'randoms/sa_15_Aleatory_{i}.csv')
    d['date'] = pd.to_datetime(d.date)
    
    
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
    
    
    modelo = 2
    
    
    for modelo in [2,3,4,5,6,7,8,9,10,11,12]:
        dtrain['cluster'] = dtrain[f'cls_{modelo}']
        for clus in dtrain.cluster.unique():
            
            X = dtrain[dtrain.cluster == clus]
            
            
            combinedHistory = X[['Mak',
                   'alpha', 'delta', 'kc', 'kcmod', 'ktmod', 'ktmod:kcmod',
                   'ktmod:alpha', 'Mak:alpha', 'kcmod:alpha', 'ktmod:Mak', 'Mak:kcmod',
                   ]]
            
            
            model = smf.glm(formula = "kc ~ Mak + alpha + delta + kcmod + ktmod +  ktmod:kcmod + ktmod:alpha + Mak:alpha + kcmod:alpha + ktmod:Mak +Mak:kcmod", 
                            data = combinedHistory, 
                            family = sm.families.Binomial())
            
            result = model.fit()# Display and interpret results
            
            Xtest = dtest[['Mak',
                   'alpha', 'delta', 'kc', 'kcmod', 'ktmod', 'ktmod:kcmod',
                   'ktmod:alpha', 'Mak:alpha', 'kcmod:alpha', 'ktmod:Mak', 'Mak:kcmod',
                   ]]
    
    
    
            dtest[f'modelo_{modelo}__cluser_{clus}'] = result.predict(Xtest) * dtest['Clear sky GHI'].values
    
    
    
    for modelo in [2,3,4,5,6,7,8,9,10,11,12]:
        dtest[f'pred_mod_{modelo}'] = np.nan
        for clus in dtrain[f'cls_{modelo}'].unique():
            dtest[f'pred_mod_{modelo}'] = np.where( dtest[f'cls_{modelo}']== clus, dtest[f'modelo_{modelo}__cluser_{clus}'], dtest[f'pred_mod_{modelo}']   )
            
        
    
        
        
        
    errormae = []
    errormbe = []
    errorrmsd = []
    errorksi = []
    errorover = []
    errorcpi = []
    for modelo in [2,3,4,5,6,7,8,9,10,11,12]:
        
        plt.figure(modelo)
        plt.plot(dtest.ghi, dtest[f'pred_mod_{modelo}'], '.')
    
        true = dtest.ghi
        pred = dtest[f'pred_mod_{modelo}']
        
        errorrmsd.append( m.rrmsd(true, pred) )
        errormbe.append( m.rmbe(true, pred) )
        errormae.append( m.rmae(true, pred) )
        
        
        ksi = det.kolmogorov_smirnov_integral(true, pred)/ true.mean() * 100
        over = det.over(true, pred)/ true.mean() * 100
        cpi = (ksi + over + 2 *m.rrmsd(true, pred))/4
        
        errorksi.append( ksi )
        errorover.append( over)
        errorcpi.append( cpi )
        # print(f"Modelo se separación: {modelo}")
        
        # print(m.rmbe(true, pred))
        # print(m.rmae(true, pred))
        # print(m.rrmsd(true, pred))
        # print("\n")
        # print("\n")
    dfmae.loc[len(dfmae)] = errormae
    dfmbe.loc[len(dfmbe)] = errormbe
    dfrmsd.loc[len(dfrmsd)] = errorrmsd
    dfksi.loc[len(dfksi)] = errorksi
    dfover.loc[len(dfover)] = errorover
    dfcpi.loc[len(dfcpi)] = errorcpi


plt.plot(dfmae.std())


dfmae.to_csv('randomMetrics/mae.csv', index=False)
dfmbe.to_csv('randomMetrics/mbe.csv', index=False)
dfksi.to_csv('randomMetrics/ksi.csv', index=False)
dfover.to_csv('randomMetrics/over.csv', index=False)
dfcpi.to_csv('randomMetrics/cpi.csv', index=False)