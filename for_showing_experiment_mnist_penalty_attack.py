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
import torchattacks

#from advertorch.test_utils import LeNet5
#from advertorch_examples.models import LeNet5Madry
from adv_lib.utils.attack_utils import run_attack
from adv_lib.attacks import alma, ddn,fmn,fab,fast_minimum_norm
from functools import partial
from adv_lib.utils.lagrangian_penalties import all_penalties

import random
from penalty_attack import penalty_attack
#from penalty_attack_momentum import penalty_attack
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


{'model':'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model': 'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model': 'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},

{'model': 'LeNet5','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.99,'loss_type':'CW'},
{'model': 'LeNet5','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.99,'loss_type':'CW'},
{'model': 'LeNet5','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.99,'loss_type':'CW'},

#{'model':'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},


#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},


#{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},


#{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},
#{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},
#{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},







#{'model': 'ddn','attack':'PenaltyL2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':5,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None},
        #{'model': 'ddn','attack':'PenaltyL2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':5.0,'out_loop_num':5,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None}, #success 0.9995 per2.66
#{'model': 'ddn','attack':'PenaltyL2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':5,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None},
        #{'model': 'ddn','attack':'PenaltyL2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':5,'inner_iter_max':300,'StepSize':0.01,'loss_type':'DLR','targeted_labels':None},
#{'model': 'trades','attack':'PenaltyL2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':5,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None},
#{'model': 'trades','attack':'PenaltyL2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':6,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None},
        #{'model': 'trades','attack':'PenaltyL2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':6,'inner_iter_max':300,'StepSize':0.01,'loss_type':'DLR','targeted_labels':None},

    ]
cnt=0
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
        for i in range(0, 10, item['batch_size']):
            print('***************{}th batch***********'.format(i))
            images=test_data_normalized[correct_index[i*item['batch_size']:i*item['batch_size']+item['batch_size']]].clone().detach()
            labels=test_labels[correct_index[i*item['batch_size']:i*item['batch_size']+item['batch_size']]]

            #for showing image
            outputs = model(images)
            _, labels_predict = torch.max(outputs, 1)
            if cnt==0:
                for i in range(10):
                    # define subplot
                    plt.subplot(7,10, cnt*10+1 + i)
                    #plt.subplot(1,10,1 + i)
                    # plot raw pixel data
                    plt.axis('off')
                    plt.imshow(images[i].cpu().data.reshape(28,28), cmap=plt.get_cmap('gray'))
                    plt.title('Predict:{}'.format(labels_predict[i].item()), fontsize='xx-small', x=0.5, y=0.90)
                    plt.subplots_adjust(wspace=0.01, hspace=0.1)
                cnt+=1
                # show the figure
                plt.show()

            success, adv_images, perturbation = penalty_attack(model, images, labels,penalty_type=item['penalty_type'],NumClass=item['NumClass'],
                     mu=item['mu'],alpha=item['alpha'],out_loop_num=item['out_loop_num'],inner_iter_max=item['inner_iter_max'],StepSize=item['StepSize'],p_norm=item['p_norm'], targeted_labels=item['targeted_labels'],beta=item['beta'],rho=item['rho'],loss_type=item['loss_type'])

            #for showing image
            outputs = model(adv_images)
            _, labels_predict = torch.max(outputs, 1)
            for i in range(10):
                # define subplot
                plt.subplot(7,10, cnt*10+1 + i)
                #plt.subplot(1,10,1 + i)
                # plot raw pixel data
                plt.axis('off')
                plt.imshow(adv_images[i].cpu().data.reshape(28,28), cmap=plt.get_cmap('gray'))
                plt.title('Predict:{}'.format(labels_predict[i].item()), fontsize='xx-small', x=0.5, y=0.90)
                plt.subplots_adjust(wspace=0.01, hspace=0.1)
            # show the figure
            plt.show()
        #    plt.subplots_adjust(wspace=0.1, hspace=0.01)

            cnt+=1
