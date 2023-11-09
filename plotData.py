#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 21:29:23 2023

@author: solar
"""

import pandas as pd


d = pd.read_csv('sa_15.csv')
d['date'] = pd.to_datetime(d.date)


vi = []

import numpy as np
for Y in d.date.dt.year.unique():
    dfil = d[d.date.dt.year==Y]
    for N in dfil.date.dt.dayofyear.unique():
        dff = dfil[dfil.date.dt.dayofyear== N]
        dff['GHIK-1'] = dff.ghi.shift(+1)
        dff['med1'] = ((dff.ghi - dff['GHIK-1']) ** 2 + 1)**0.5
        dff['GHIargpK-1'] = dff['Clear sky GHI'].shift(+1)
        dff['med2'] = ((dff['Clear sky GHI'] - dff['GHIargpK-1']) ** 2 + 1)**0.5
        
        dff['vi'] = np.where(dff.med1.isna() | dff.med1.isna() ,np.nan,dff.med1/dff.med2  )
        
        if len(dff.vi.dropna()>0):
            dff['vi'] = dff.vi/max(dff.vi.dropna()) 
        
        for x in dff.vi:
            vi.append(x)


d['vi'] = vi


d = d[d.kc<1.3]



import matplotlib.pyplot as plt

plt.plot(d.vi, d.kc, '.r', markersize=0.4)



import statsmodels.formula.api as smf
import statsmodels.api as sm

dt = d[int(len(d)/2):]

dt = dt.dropna()

for n in range(2, 11):


    X = np.array(dt[["kc","vi"]])
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n ).fit(X)
    centroids = kmeans.cluster_centers_
    
    
    # Predicting the clusters
    labels = kmeans.predict(X)
    # Getting the cluster centers
    C = kmeans.cluster_centers_
    colores=['red','green','blue','pink', 'yellow','orange',
             'black','gray', 'cyan', 'brown']
    asignar=[]
    pre_clase=[]
    for row in labels:
        asignar.append(colores[row])
        pre_clase.append(row)

    
    dt[f'cluster_{n}'] = labels
    plt.figure(n)
    plt.scatter(dt.vi, dt.kc, c=asignar)
    plt.show()
    
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.datasets import make_classification
import joblib
from sklearn.ensemble import RandomForestClassifier



for n in range(2, 11):
    rf = RandomForestClassifier()
    dtrain = dt[:int(len(dt)/2)]
    rf.fit(dtrain[['Mak','delta','kcmod', 'ktmod','kc' ]], dtrain[f'cluster_{n}']  )    
    
    dtest = dt[int(len(dt)/2):]
    predictions = rf.predict(dtest[['Mak','delta','kcmod', 'ktmod','kc' ]])
    cm = confusion_matrix(dtest[f'cluster_{n}'], predictions, labels=rf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=rf.classes_)
    
    disp.plot()
    


dt = dt[['kt','Mak','delta','kcmod', 'ktmod','kc' ]].dropna()



dt = dt[dt.kt>0]

dt = dt[dt.kc>0.5]


combinedHistory = dt

model = smf.glm(formula = "kc ~ kcmod + Mak + delta + ktmod", 
                data = combinedHistory, 
                family = sm.families.Binomial())
# Fit the model
result = model.fit()# Display and interpret results
print(result.summary())# Estimated default probabilities

test = d[: int(105120/2)]
test = test[test.kc>0.5]
predictions = result.predict(test[['kcmod','Mak','delta','ktmod' ]].dropna())



plt.plot(test.ghi.values, predictions*test['Clear sky GHI'].values, '.r')


test['pred'] = predictions*test['Clear sky GHI'].values


plt.plot(test.index, test.ghi, '-r', label="med")
plt.plot(test.index, test.GHI, '-b', label="cams")
plt.plot(test.index, test.pred, '-g', label="pred")
plt.legend()

