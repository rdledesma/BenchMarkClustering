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

d = pd.read_csv('sa_diego.csv')
d['date'] = pd.to_datetime(d.date)
c = pd.read_csv('sa_15_cony.csv')
c['date'] = pd.to_datetime(c.date)


d = (d.set_index('date')
      .reindex(c.date)
      .rename_axis(['date'])
      #.fillna(0)
      .reset_index())

c['cluster'] = d.clsDiego.values
c = c[c.alpha>10]
c = c[c.kc<1.3]
c = c.dropna()

train = c[c.date.dt.year == 2015]
test = c[c.date.dt.year < 2015]

for clus in c.cluster.unique():
    dtrain = train[train.cluster == clus]
    
    combinedHistory = dtrain[['Mak',
           'alpha', 'delta', 'kc', 'kcmod', 'ktmod', 'ktmod:kcmod',
           'ktmod:alpha', 'Mak:alpha', 'kcmod:alpha', 'ktmod:Mak', 'Mak:kcmod',
           ]]

    model = smf.glm(formula = "kc ~ Mak + alpha + delta + kcmod + ktmod +  ktmod:kcmod + ktmod:alpha + Mak:alpha + kcmod:alpha + ktmod:Mak +Mak:kcmod", 
                    data = combinedHistory, 
                    family = sm.families.Binomial())

    # Fit the model
    result = model.fit()# Display and interpret results



    X = test[['Mak',
           'alpha', 'delta', 'kc', 'kcmod', 'ktmod', 'ktmod:kcmod',
           'ktmod:alpha', 'Mak:alpha', 'kcmod:alpha', 'ktmod:Mak', 'Mak:kcmod',
           ]]

    
    test[f'pred{clus}']= result.predict(X) * test['Clear sky GHI']






test['pred'] = np.nan

for clus in c.cluster.unique():
    test['pred'] = np.where(test.cluster == clus, test[f'pred{clus}'], test.pred)




print(result.summary())# Estimated default probabilities



plt.plot(test.ghi, test.pred, '.')



true = test.ghi
pred = test.pred

m.rmbe(true, pred)
m.rmae(true, pred)
m.rrmsd(true, pred)
