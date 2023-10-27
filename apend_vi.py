#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:24:37 2023

@author: dario
"""

import pandas as pd
import numpy as np


d = pd.read_csv('sa_15.csv')
d = d.drop(['Unnamed: 0'], axis=1)

d['date'] = pd.to_datetime(d.date)
d['f0'] = (d.alpha>7) & (d.kc<1.3)
d = d[d.f0]
d = d.dropna()
vi = []

for Y in d.date.dt.year.unique():
    dfil = d[d.date.dt.year==Y]
    for N in dfil.date.dt.dayofyear.unique():
        dff = dfil[dfil.date.dt.dayofyear == N]
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

dna = d.dropna()




from sklearn.cluster import KMeans    
for i in range(2,11):
    


    
    X = dna[[ 'kc', 'vi']]
    kmeans = KMeans(n_clusters=i).fit(X)
    
    centroids = kmeans.cluster_centers_
    
    
    # Predicting the clusters
    labels = kmeans.predict(X)
    # Getting the cluster centers
    C = kmeans.cluster_centers_
    colores=['red','green','blue','gray','yellow']
    asignar=[]
    pre_clase=[]
    for row in labels:
        #asignar.append(colores[row])
        pre_clase.append(row)
    
     
    dna[f'cluster_{i}'] = pre_clase
    	
    # # Getting the values and plotting it
    # f1 = dna['kc'].values
    # f2 = dna['vi'].values
    
    # import matplotlib.pyplot as plt
    # plt.figure(i)
    # plt.scatter(f2,f1, c=asignar, s=1)






