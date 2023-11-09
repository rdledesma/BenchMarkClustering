# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:51:46 2023

@author: Cony
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#%%

ruta = 'sa_15_cony.csv'
df = pd.read_csv(ruta)
df['date'] = pd.to_datetime(df.date)
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# Aplico filtros 
df = df[df.CTZ > 0]  # por noche
df = df[df.kc < 1.3] # por sobreirradiancia 
df = df[df.alpha > 10] # por horizonte

# Agrupar los datos por día
dat_diario = df.groupby(df['date'].dt.date)
df_diarios = dat_diario.mean(numeric_only=True)

#%%
kc_ref = 0.85

'''
Tipo de cielo = sky
    sky0 = cielo nublado
    sky1 = cielo despejado
'''

df_diarios['sky'] = df_diarios['kc'].apply(lambda x: 1 if x >= kc_ref else 0)

df_diarios['sky_mod'] = df_diarios['kcmod'].apply(lambda x: 1 if x >= kc_ref else 0)

df_diarios= df_diarios.reset_index()
df_diarios['date'] = pd.to_datetime(df_diarios.date)


# Etiquetar los datos de df según la etiqueta de df_diarios
df['sky'] = df['date'].dt.date.isin(df_diarios[df_diarios['sky'] == 1]['date'].dt.date).astype(int)
df['sky_mod'] = df['date'].dt.date.isin(df_diarios[df_diarios['sky_mod'] == 1]['date'].dt.date).astype(int)

#%%
df_sky0 = df[df['sky'] == 0]
df_sky1 = df[df['sky'] == 1]

df_sky0_mod = df[df['sky_mod'] == 0]
df_sky1_mod = df[df['sky_mod'] == 1]

#%%

plt.figure(1)
plt.plot(df.CTZ, df.ghi, '.g', alpha=0.5, markersize=15)
plt.plot(df[df.sky == 0].CTZ, df[df.sky == 0].ghi, '.b')
plt.plot(df[df.sky == 1].CTZ, df[df.sky == 1].ghi, '.r')
plt.show()

plt.figure(2)
plt.plot(df.CTZ, df.GHI, '.g', alpha=0.5, markersize=15)
plt.plot(df[df.sky_mod == 0].CTZ, df[df.sky_mod == 0].GHI, '.b')
plt.plot(df[df.sky_mod == 1].CTZ, df[df.sky_mod == 1].GHI, '.r')
plt.show()

#%%
df.to_csv('sa_15_cony_V2.csv', index=False)