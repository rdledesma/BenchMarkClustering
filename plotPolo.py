#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:06:08 2023

@author: dario
"""

import pandas as pd


d = pd.read_csv('sa_15_polo.csv')



d['date'] = pd.to_datetime(d.date)
d = d[d.date.dt.year == 2015]

d['HR'] = d['date'].dt.hour + d['date'].dt.minute/60 + d['date'].dt.second/3600




d0 = d[d.date.dt.dayofyear == 160]
d1 = d[d.date.dt.dayofyear == 12]
d2 = d[d.date.dt.dayofyear == 36]
d3 = d[d.date.dt.dayofyear == 2]






import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

# plt.plot(d0.HR  ,d0.GHI.values, label="Overcast")
ax.plot(d0.HR , d0.GHI.values ,label="Clear", color="blue")
ax.plot(d2.HR , d2.GHI.values ,label="Overcast", color="red")
ax.plot(d1.HR , d1.GHI.values ,label="Overcast", color="green")
ax.plot(d3.HR , d3.GHI.values ,label="Overcast", color="gray")
plt.legend()
ax.set_xlabel('Hour of the day')
ax.set_ylabel('GHI Irradiance (W/m²)')
plt.xlim(left=10.75, right = 22.4)  

