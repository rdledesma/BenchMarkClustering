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
from solarforecastarbiter import metrics

d = pd.read_csv('sa_15_Diego2.csv')
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
c = c[c.ghi>5]
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

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# Crear datos de ejemplo
x = test.ghi
y = test.pred

# Calcular la densidad de puntos
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# Configuración de la paleta de colores
cmap = sns.color_palette("viridis", as_cmap=True)

# Crear el gráfico de dispersión con densidad bidimensional
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=z, s=10, cmap=cmap)

# Añadir barra de color
cbar = plt.colorbar()
cbar.set_label('Density')
cbar.set_ticks([])
# Personalizar el gráfico
#plt.title('Gráfico de Dispersión con Densidad Bidimensional')
plt.xlabel('Measured ghi (W/m²)')
plt.ylabel('Site adapted ghi (W/m²)')
plt.xlim(0, 1300)  # Reemplaza 0 y 100 con tus límites deseados para el eje x
plt.ylim(0, 1200)  # Reemplaza 0 y 100 con tus límites deseados para el eje y
plt.plot([x for x in range(1200)], '--', color="black" )
plt.grid(True)
# Mostrar el gráfico
plt.show()








true = test.ghi
pred = test.pred

print(m.rmbe(true, pred))
print(m.rmae(true, pred))
print(m.rrmsd(true, pred))
r = m.rrmsd(true, pred)

ksi = metrics.deterministic.kolmogorov_smirnov_integral(true, pred)/ true.mean() * 100



over = metrics.deterministic.over(true, pred) / true.mean() * 100

cpi = (ksi + over + 2 *r)/4



