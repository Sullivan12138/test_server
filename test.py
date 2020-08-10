import time
import datetime
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import LassoCV
import json

a = 'aa'

with open('./data/requests2.log', 'r') as f:
    load_dict = json.load(f)
    for i in load_dict:
        for metric in i:
            print(metric['metric']['instance'], '-', metric['metric']['type'], ':', metric['values'][-1][1])

