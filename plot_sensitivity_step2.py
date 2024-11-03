import torch
import numpy as np
import matplotlib.pyplot as plt

data=torch.load('./list_for_plot_sensitivity.pt')
# 生成一些示例数据
x = [-3,-2,-1,0,1,2,3]
LeNet1 = data['lenet_asr']
LeNet2 = data['lenet_pert']
standard1 = data['standard_asr']
standard2 = data['standard_pert']
ddn1 = data['ddn_asr']
ddn2 = data['ddn_pert']

# 创建一个图形和两个y轴
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# 绘制折线图
line_lenet1 = ax1.plot(x, LeNet1, label='LeNet5-M_ASR', color='royalblue', marker='o', ls='-.')
line_lenet2 = ax2.plot(x, LeNet2, label='LeNet5-M_Pert', color='tomato', marker=None, ls='--')
line_standard1 = ax1.plot(x, standard1, label='standard-M_ASR', color='green', marker='o', ls='-.')
line_standard2 = ax2.plot(x, standard2, label='standard-M_Pert', color='y', marker=None, ls='--')
line_ddn1 = ax1.plot(x, ddn1, label='DDN-M_ASR', color='steelblue', marker='o', ls='-.')
line_ddn2 = ax2.plot(x, ddn2, label='DDN-M_Pert', color='coral', marker=None, ls='--')
ax1.text(0, 0.03, '0.714', fontsize=12)
ax1.text(0, 0.15, '1.401', fontsize=12)
ax1.text(0.8, 0.35, '2.697', fontsize=12)
# 设置x轴和y轴的标签，指明坐标含义
ax1.set_xlabel('log$_{10}\mu$', fontdict={'size': 10})
ax1.set_ylabel('ASR', fontdict={'size': 10})
ax1.legend(bbox_to_anchor=(0.3,1),fontsize='small')
ax2.set_ylabel('Pert.', fontdict={'size': 10})
ax2.legend(bbox_to_anchor=(1,0.9),fontsize='small')
#添加图表题
plt.title('Sensitivity of Penalty Parameter')
# 添加图例
print('*********')


