# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:02:23 2021

@author: shiuli Subhra Ghosh
"""
import pandas as pd

data = pd.DataFrame()

data['X'] = [10,20]
data['Y'] = [20,50]
data['Z'] = [30,60]


df = data[['X','Y']]


