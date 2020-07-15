# 代码说明
我修改的部分主要包括predict_tikv.py和prome.py、test_prome.py。

## predict_tikv.py
在这个文件中，我加入了特征选择的功能，并修改了程序的流程。
现在程序的流程变成了先获取数据，获取之后存储，
然后对获取到的数据进行特征选择，选择合适的特征数据再进行训练，
训练好后保存模型。再利用已经训练好的模型预测数据。
预测时输入的数据也只需要筛选出的特征对应的数据即可。

现在的特征有12个，分别是：
'cpu', 'io util', 'io latency', 'grpc cpu', 'io bandwidth',
 'grpc duration put', 'grpc duration put prewrite', 
 'grpc duration put commit', 'grpc duration put cop', 
 'grpc duration put get', 'grpc duration putscan'

我的特征选择主要做了以下工作：
- 计算每个特征数据的方差，剔除了那些变化不大的数据
- 计算Pearson相关系数，剔除相关性过高的特征
- 使用LightGBM给每个特征打分，剔除重要性为0的特征
- 剔除重要性小于阈值的特征
- 进行l1正则化

我将原来的代码中边获取数据边预测边训练的流程改成先获取数据，
再预测，再训练。现在的程序训练数据是：先获取40个batch的数据，
每个batch中包含20个训练数据集，每个数据集是15分钟的特征数据。
目标数据则是这15分钟之后的5分钟的cpu usage。
这样，每次训练时，训练40个循环，每个循环训练20批数据。

同时，我将原来代码中单输入的情况修改为多输入。

## prome.py
我添加了一个文件prome.py，在其中添加了获取我们需要选择的特征
数据的函数。

## test_prome.py
这个文件是测试获取数据能否正常获取的数据。我在这个文件中写了
尝试获取新特征数据的代码，用于测试获取特征数据的函数能否正常获取
到数据。
            