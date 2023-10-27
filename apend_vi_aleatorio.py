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
dna = dna[dna.f0]



p95 = dna['vi'].quantile(0.95)

dna['cc'] = dna['vi'] > p95
dna['vin'] = dna.vi/max(dna.vi)


import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(dna.vin, dna.kc, '.b')
plt.plot(dna[dna.cc].vin, dna[dna.cc].kc, '.r')
plt.plot()









len((dna[dna.cc].vin)) / len((dna))  * 100



import pandas as pd
from sklearn.neighbors import KernelDensity
# Supongamos que 'df' es tu DataFrame y tiene columnas 'vi' y 'kc'
# X será una matriz con las columnas 'vi' y 'kc' como características
X = dna[['vi', 'kc']]


kde = KernelDensity(bandwidth=0.01)  # Puedes ajustar el ancho de banda según sea necesario
kde.fit(X)

log_densidades=  kde.score_samples(X[:])



# Normaliza los logaritmos de las densidades a valores entre 0 y 1
densidades = np.exp(log_densidades)
densidades = (densidades - densidades.min()) / (densidades.max() - densidades.min())


dna['densidad'] = densidades

# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(dna.vi, dna.kc, dna.densidad, c='black', s=0.1)

my_centroids = np.array([[ 0.94, 0.025, 0.69], [0.912, 0.0156,  0.025], 
                         [ 0.766, 0.937, 0.007],
                         [ 0.15, 0.01, 0.07], [ 0.735, 0.539, 0.005]])




X = dna[[ 'kc', 'vi', 'densidad']]

from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, init=my_centroids ).fit(X)

centroids = kmeans.cluster_centers_


# Predicting the clusters
labels = kmeans.predict(X)
# Getting the cluster centers
C = kmeans.cluster_centers_
colores=['red','green','blue','gray','yellow']
asignar=[]
pre_clase=[]
for row in labels:
    asignar.append(colores[row])
    pre_clase.append(row)

 

	
# Getting the values and plotting it
f1 = dna['kc'].values
f2 = dna['vi'].values

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(f2,f1, c=asignar, s=1)

