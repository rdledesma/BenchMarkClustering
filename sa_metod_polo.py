# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:22:53 2023



@author: Cony
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Metrics as ms
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


# df sin nan
ruta = 'D:/GEERS2023/On_2023/ProyectosDario_/Proyecto_01/sa_15_cony_V2.csv'
df = pd.read_csv(ruta)
df['date'] = pd.to_datetime(df.date)


df = df[df.ghi > 5]

#dtest = df.dropna()

# datos de testeo: los correspondientes al año 2015
dtest = df[df.date.dt.year < 2015] 

# Variable 
X = dtest[['Mak', 'alpha', 'delta', 'kcmod', 'ktmod', 'ktmod:kcmod', 'ktmod:alpha', 'Mak:alpha', 'kcmod:alpha', 'ktmod:Mak', 'Mak:kcmod']]

dtrain = df[df.date.dt.year == 2015]

# Separacion de datos
for c in df.sky_mod.unique(): 
    d = dtrain[dtrain.sky_mod == c ]
    
    model = smf.glm(formula = "kc ~ Mak + alpha  + delta + kcmod + ktmod + ktmod:kcmod + ktmod:alpha + Mak:alpha + kcmod:alpha + ktmod:Mak + Mak:kcmod", 
    data = d, 
    family = sm.families.Binomial())
    # Fit the model
    result = model.fit() # entrenaiento y guardado
     
        # Create a linear regression model
    Xr = d[['Mak', 'alpha', 'delta', 'kcmod', 'ktmod', 'ktmod:kcmod', 'ktmod:alpha', 'Mak:alpha', 'kcmod:alpha', 'ktmod:Mak', 'Mak:kcmod']]    
    modelR = LinearRegression(fit_intercept=True)
    modelR.fit(Xr, d.kc)
        
    predictions = dtest['Clear sky GHI'] * result.predict(X) # reconstruccion
    predictionsR = dtest['Clear sky GHI'] * modelR.predict(X) 
    
    dtest[f'ghiPredR{c}'] = predictionsR  # rec = rectificate
    dtest[f'ghiPred{c}'] = predictions # rec = rectificate
    
dtest ['ghiPred']  = np.nan

dtest['ghiPredR'] = np.where(dtest.sky_mod == 0, dtest.ghiPredR0, dtest.ghiPredR1)

dtest['ghiPred'] = np.where(dtest.sky_mod == 0, dtest.ghiPred0, dtest.ghiPred1)


  #%%

true = dtest.dropna().ghi
pred_sat = dtest.dropna().GHI
pred_adap = dtest.dropna().ghiPred
pred_adapR = dtest.dropna().ghiPredR

ms.r_mean_bias_error(true, pred_sat)
ms.r_mean_bias_error(true, pred_adap)
ms.r_mean_bias_error(true, pred_adapR)

ms.r_mean_absolute_error(true, pred_sat)
ms.r_mean_absolute_error(true, pred_adap)
ms.r_mean_absolute_error(true, pred_adapR)

ms.rrmsd(true, pred_sat)
ms.rrmsd(true, pred_adap)  
ms.rrmsd(true, pred_adapR) 

#%% 



plt.figure(10)
plt.plot(dtest.CTZ, dtest.ghi, '.g')
plt.plot(dtest.CTZ, dtest.GHI, '.r')

plt.figure(11)
plt.plot(dtest.CTZ, dtest.ghi, '.g')
plt.plot(dtest.CTZ, dtest.ghiPred, '.b')


plt.figure(12)
plt.plot(dtest.ghi, dtest.ghi, 'g')

plt.plot(dtest.ghi, dtest.GHI, '.r')

plt.figure(13)
plt.plot(dtest.ghi, dtest.ghiPred0, '.b')
plt.figure(14)
plt.plot(dtest.ghi, dtest.ghiPred1, '.r')

plt.plot(dtest.ghi, dtest.ghiPred, '.r')
