import time
import datetime
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import LassoCV
import json

a = ['cc', 'bb']
b = json.loads(a)
with open('test.txt', 'a+') as f:
    f.write(b)
    f.write(b)
