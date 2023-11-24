#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:11:12 2023

@author: solar
"""

import pandas as pd
d = pd.read_csv('sa_15_vimod.csv')
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




