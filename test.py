import time
import datetime
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import LassoCV

X = np.array([[ 0.87, -1.34,  0.31 ] for _ in range(750)])
df = pd.DataFrame(X)
print(df.shape)
y = [0 for _ in range(300)]
for _ in range(375):
      y.append(1.1)
for _ in range(75):
      y.append(2.2)
y = np.array(y)
print(y.shape)
# clf = LogisticRegression(penalty='l1', solver='liblinear')

# clf.fit(X, y.astype(int))
# selector = SelectFromModel(estimator=clf)
# selector.fit(X, y.astype(int))
# print(selector.estimator_.coef_)
# print(selector.threshold_)
# print(selector.get_support())
# print(selector.fit_transform(X, y.astype(int)))

clf = LassoCV()
clf.fit(X,y)

selector = SelectFromModel(clf, prefit=True)
support = selector.get_support()
print(support[1])