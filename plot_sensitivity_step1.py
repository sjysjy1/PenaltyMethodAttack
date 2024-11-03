import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model_MNIST import model_MNIST
from LeNet5 import Model_LeNet5
import time
import matplotlib.pyplot as plt


import random
from penalty_attack import penalty_attack

def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out


seed_torch()
device=('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

test_dataset=datasets.MNIST(root='./data',train=False,download=True,transform=transforms.ToTensor())

criterion=nn.CrossEntropyLoss()
batch_size=256
list_para=[
{'model':'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.001,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model':'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.01,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model':'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model':'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model':'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model':'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':100,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model':'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':1000,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},

{'model':'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.001,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model':'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.01,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model':'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model':'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model':'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model':'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':100,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model':'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':1000,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},

{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.001,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.01,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':100,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':1000,'alpha':10.0,'out_loop_num':1,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},



#{'model':'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},


#{'model': 'LeNet5','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.99,'loss_type':'CW'},
#{'model': 'LeNet5','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.99,'loss_type':'CW'},
#{'model': 'LeNet5','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.99,'loss_type':'CW'},

#{'model':'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},


#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},


#{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},
#{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},
#{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},


#{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},


#{'model': 'ddn','attack':'PenaltyL2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':5,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None},
        #{'model': 'ddn','attack':'PenaltyL2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':5.0,'out_loop_num':5,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None}, #success 0.9995 per2.66
#{'model': 'ddn','attack':'PenaltyL2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':5,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None},
        #{'model': 'ddn','attack':'PenaltyL2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':5,'inner_iter_max':300,'StepSize':0.01,'loss_type':'DLR','targeted_labels':None},
#{'model': 'trades','attack':'PenaltyL2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':5,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None},
#{'model': 'trades','attack':'PenaltyL2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':6,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None},
        #{'model': 'trades','attack':'PenaltyL2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':6,'inner_iter_max':300,'StepSize':0.01,'loss_type':'DLR','targeted_labels':None},

    ]
plot_list_asr_lenet  =[]
plot_list_pert_lenet =[]
plot_list_asr_standard =[]
plot_list_pert_standard =[]
plot_list_asr_ddn =[]
plot_list_pert_ddn =[]
for item in list_para:
    list_success_fail=[]
    list_pert=[]
    list_iterNum=[]
    print(item)
    if item['model']=='LeNet5':
        model = Model_LeNet5()
        model.load_state_dict(torch.load('./MNIST_LeNet_onlyweight.pth'), False)
    elif item['model']=='mnist-L2-MMA-4.0-sd0':
        model = LeNet5Madry()
        model = Model_LeNet5()
        #model.load_state_dict(torch.load('./mnist-L2-MMA-4.0-sd0-model_best.pt'), False)
        model=LeNet5()
        model.load_state_dict(torch.load('./MNIST_LeNet_onlyweight.pth'), False)
        #model.load_state_dict(torch.load('./mnist_lenet5_advtrained.pt'), False)

    else:
        model = model_MNIST()
        if item['model'] == 'Standard':
            #model.load_state_dict(torch.load('./MNIST_MODEL_CW_selftrain.pth'), False)
            model.load_state_dict(torch.load('./models/mnist/mnist_regular.pth'), False)
        elif item['model'] == 'ddn':
            model.load_state_dict(torch.load('./models/mnist/mnist_robust_ddn.pth'), False)  # ALM paper github : l2 adversarially trained
        elif item['model'] == 'trades':
            model.load_state_dict(torch.load('./models/mnist/mnist_robust_trades.pt'),False)  # ALM paper github: l_\infty adversarially trained
    #model.load_state_dict(torch.load('../models/mnist/model_mnist_smallcnn.pt'), False)  # model from github of trades:https://github.com/yaodongyu/TRADES
    model.to(device)
    model.eval()  # turn off the dropout
    test_data = torch.unsqueeze(test_dataset.data, dim=1)
    test_labels = test_dataset.test_labels.to(device)
    test_data_normalized = test_data / 255.0
    test_data_normalized = test_data_normalized.to(device)
    outputs = model(test_data_normalized)
    _, labels_predict = torch.max(outputs, 1)
    correct = torch.eq(labels_predict, test_labels)
    correct_sum = correct.sum()
    correct_index=[]
    for i in range(10000):
        if correct[i]:
            correct_index.append(i)
    #print(correct.sum())
    print('clean accuracy is:', correct.sum() / 10000.0)
    start_time = time.time()
    if item['attack'] == 'PenaltyAttack':
        for i in range(correct_sum//item['batch_size']):
            print('***************{}th batch***********'.format(i))
            images=test_data_normalized[correct_index[i*item['batch_size']:i*item['batch_size']+item['batch_size']]].clone().detach()
            labels=test_labels[correct_index[i*item['batch_size']:i*item['batch_size']+item['batch_size']]]

            success, adv_images, perturbation = penalty_attack(model, images, labels,penalty_type=item['penalty_type'],NumClass=item['NumClass'],
                     mu=item['mu'],alpha=item['alpha'],out_loop_num=item['out_loop_num'],inner_iter_max=item['inner_iter_max'],StepSize=item['StepSize'],p_norm=item['p_norm'], targeted_labels=item['targeted_labels'],beta=item['beta'],rho=item['rho'],loss_type=item['loss_type'])

            list_success_fail=list_success_fail+torch.squeeze(success,dim=1).tolist()
            #list_pert=list_pert+perturbation.tolist()
            list_pert = list_pert + torch.squeeze(perturbation, dim=1).tolist()
            #list_iterNum.append(cnt)
            print('perturbation is: ',torch.t(perturbation))
            print('avg_pert is: ', perturbation.sum() / len(perturbation))
        #The last batch
        if correct_sum%item['batch_size']!=0:
            i=i+1
            print('***************{}th batch***********'.format(i))
            images = test_data_normalized[correct_index[i * item['batch_size']:]].clone().detach()
            labels = test_labels[correct_index[i * item['batch_size']:]]
            success, adv_images, perturbation= penalty_attack(model, images, labels,penalty_type=item['penalty_type'],NumClass=item['NumClass'],
                      mu=item['mu'], alpha=item['alpha'],out_loop_num=item['out_loop_num'],inner_iter_max=item['inner_iter_max'],StepSize=item['StepSize'],p_norm=item['p_norm'],targeted_labels=item['targeted_labels'])
            list_success_fail=list_success_fail+torch.squeeze(success,dim=1).tolist()
            #list_pert=list_pert+perturbation.tolist()
            list_pert = list_pert + torch.squeeze(perturbation, dim=1).tolist()
            #list_iterNum.append(cnt)
            print('perturbation is: ', torch.t(perturbation))
            print('avg_pert is: ', perturbation.sum() / len(perturbation))
        end_time = time.time()
    time_used=end_time-start_time
    print('running time:',end_time-start_time,'seconds')
    attack_success_rate=sum(list_success_fail)/correct_sum
    print('attack success rate:',attack_success_rate)
    list_pert_success=[ pert  for pert,result in zip(list_pert,list_success_fail) if result]
    #temp=len(list_pert[list_success_fail])
    #temp1=len(list_pert)
    #temp2=sum(list_pert[list_success_fail])
    #temp3=sum(list_pert)
    avg_pert=sum(list_pert_success)/len(list_pert_success)
    print('avg_pert is',avg_pert)
    if item['model']=='LeNet5':
        plot_list_asr_lenet.append(attack_success_rate.item())
        plot_list_pert_lenet.append(avg_pert)
    elif item['model'] == 'Standard':
        plot_list_asr_standard.append(attack_success_rate.item())
        plot_list_pert_standard.append(avg_pert)
    elif item['model'] == 'ddn':
        plot_list_asr_ddn.append(attack_success_rate.item())
        plot_list_pert_ddn.append(avg_pert)
dict_save={'lenet_asr':plot_list_asr_lenet,'lenet_pert':plot_list_pert_lenet,
           'standard_asr':plot_list_asr_standard,'standard_pert':plot_list_pert_standard,
           'ddn_asr':plot_list_asr_ddn,'ddn_pert':plot_list_pert_ddn}
torch.save(dict_save,'./list_for_plot_sensitivity.pt')
print('********')





