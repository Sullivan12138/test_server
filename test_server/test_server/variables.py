"""
这个文件里包含的是训练和预测需要用到的变量
"""
input_size = 1  # 输入维度
output_size = 1  # 输出维度
rnn_unit = 12  # 隐藏层节点
lr = 0.0004  # 学习率
batch_size = 20  # 每次训练的一个批次的大小
time_step = 15  # 前time_step步来预测下一步
predict_step = 5  # 预测predict_step分钟后的负载
kp = 1  # dropout保留节点的比例
smooth = 0  # 为1则在时间维度上平滑数据
train_time = 50  # 所有数据的训练轮次
save_model_path_requests = './save/requests/'  # checkpoint存在的目录
save_model_path_cpu = './save/cpu/'
save_model_name = 'MyModel'  # saver.save(sess, './save/MyModel') 保存模型
yaml_path = '/data2/hust_tmp/cluster/tidb-cluster.yaml'
min_scale_interval = 5
weight = [1, 1, 1, 1]  # read, update, scan, insert
scan_per_cop = [0.0157, 0.0252, 0.0501]  # kv_cop与scan的比例系数
scan_per_cop_level = 0  # 比例系数的等级，0表示maxscanlength=1000，1表示100，2表示10
three_nodes_endurable_ops = 75
four_nodes_endurable_ops = 83
