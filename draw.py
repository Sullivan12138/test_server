import matplotlib.pyplot as plt
import json
import numpy as np

with open('y.json', 'r') as f:
    y = json.load(f)
    print(np.array(y).shape)
with open('y2.json', 'r') as f:
    Y = json.load(f)
    print(np.array(Y).shape)
with open('p.json', 'r') as f:
    p = json.load(f)
    p0, p1, p2, p3 = [], [], [], []
    for i in range(len(p)):
        p0.append(p[i][0])
        p1.append(p[i][1])
        p2.append(p[i][2])
        p3.append(p[i][3])

# 画图表示结果
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('time/min')
ax1.set_ylabel('proportion')
ax1.set_title('command proportion')
# x = list(range(len(y)))
# X = list(range(len(Y)))
# ax1.plot(x, y, color='b', label='real')
# ax1.plot(X, Y, color='r', label='predict')
x0 = list(range(len(p0)))
x1 = list(range(len(p1)))
x2 = list(range(len(p2)))
x3 = list(range(len(p3)))
ax1.plot(x0, p0, color='b', label='read')
ax1.plot(x1, p1, color='r', label='scan')
ax1.plot(x2, p2, color='g', label='update')
ax1.plot(x3, p3, color='y', label='insert')
plt.legend()
plt.show()