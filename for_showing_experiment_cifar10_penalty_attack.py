import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import torchattacks
from adv_lib.utils.attack_utils import run_attack
from adv_lib.attacks import alma, ddn, fmn,fab
from functools import partial
from adv_lib.utils.lagrangian_penalties import all_penalties
import random
from penalty_attack import penalty_attack
#from penalty_attack_momentum import penalty_attack
#from penalty_attack_Adam import penalty_attack
from robustbench import  load_model

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
seed_torch()
device=('cuda:0' if torch.cuda.is_available() else 'cpu')
#device='cpu'
print(device)

test_dataset=datasets.CIFAR10(root='./data',train=False,download=False,transform=transforms.ToTensor())
criterion=nn.CrossEntropyLoss()
batch_size=10
list_para=[



#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.01,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.1,'loss_type':'CW','Use_RMS':True},
#         #{'model':'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':5,'inner_iter_max':200,'StepSize':0.01,'targeted_labels':None},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':0.01,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.1,'loss_type':'CW','Use_RMS':True},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.01,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.1,'loss_type':'CW','Use_RMS':True},
#
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.01,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.0001,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW'},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':0.01,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.0001,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW'},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.01,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.0001,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW'},


#{'model': 'WongLinf','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':True},
#{'model': 'WongLinf','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':True},
#{'model': 'WongLinf','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':True},
#{'model': 'WongLinf','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW'},
#{'model': 'WongLinf','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW'},
#{'model': 'WongLinf','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW'},


{'model': 'WangL2WRN-28-10','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':True},
{'model': 'WangL2WRN-28-10','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':True},
{'model': 'WangL2WRN-28-10','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':True},

{'model': 'WangL2WRN-28-10','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW'},
{'model': 'WangL2WRN-28-10','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW'},
{'model': 'WangL2WRN-28-10','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW'},

    ]
cnt=0
for item in list_para:
    list_success_fail=[]
    list_pert=[]
    list_iterNum=[]
    print(item)
    test_data = torch.tensor(test_dataset.data, device=device)
    test_labels = torch.tensor(test_dataset.targets, device=device)
    test_data_normalized = test_data / 255.0
    #test_data_normalized = test_data_normalized
    test_data_normalized = test_data_normalized.permute(0, 3, 1, 2)
    if item['model'] == 'Standard':
        model = load_model(model_name='Standard', dataset='cifar10', norm='Linf')
    elif item['model'] == 'WongLinf':
        model = load_model(model_name='Wong2020Fast', norm='Linf')
    elif item['model'] == 'WangL2WRN-28-10':
        #model = load_model(model_name='Wang2023Better_WRN-70-16', dataset='cifar10', norm='L2')Wang2023Better_WRN-28-10
        model = load_model(model_name='Wang2023Better_WRN-28-10', dataset='cifar10', norm='L2')
    model.to(device)
    model.eval()  # turn off the dropout
    test_accuracy = False
    if test_accuracy == True:
        predict_result = torch.tensor([], device=device)
        for i in range(500):
            outputs = model(test_data_normalized[20 * i:20 * i + 20])
            _, labels_predict = torch.max(outputs, 1)
            predict_result = torch.cat((predict_result, labels_predict), dim=0)
        correct = torch.eq(predict_result, test_labels)
        torch.save(correct, './result/cifar10-first1000/{}_Cifar_correct_predict.pt'.format(item['model']))
    else:
        correct = torch.load('./result/cifar10-first1000/{}_Cifar_correct_predict.pt'.format(item['model']))
    correct_sum = correct.sum()
    clean_accuracy = correct_sum / 10000.0
    correct_sum = correct[0:1000].sum()
    print('model clean accuracy:', clean_accuracy)
    correct_index=[]
    for i in range(1000):
        if correct[i]:
            correct_index.append(i)

    start_time = time.time()
    if item['attack'] == 'PenaltyAttack':
        for i in range(0, item['batch_size'], item['batch_size']):
            print('***************{}th batch***********'.format(i))
            images=test_data_normalized[correct_index[i*item['batch_size']:i*item['batch_size']+item['batch_size']]].clone().detach()
            labels=test_labels[correct_index[i*item['batch_size']:i*item['batch_size']+item['batch_size']]]
            #for showing image
            outputs = model(images)
            _, labels_predict = torch.max(outputs, 1)
            if cnt==0:
                for i in range(10):
                    # define subplot
                    plt.subplot(7,item['batch_size'], cnt*10+1 + i)
                    #plt.subplot(1,10,1 + i)
                    # plot raw pixel data
                    plt.axis('off')
                    plt.imshow(np.transpose(torchvision.utils.make_grid(images[i].cpu().data, normalize=True),(1,2,0)))
                    plt.title('Predict:{}'.format(labels_predict[i].item()), fontsize='xx-small', x=0.5, y=0.9)
                    plt.subplots_adjust(wspace=0.01, hspace=0.1)
                # show the figure
                plt.show()
                #plt.subplots_adjust(wspace=-0.4, hspace=0.01)
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
                cnt += 1
            if item['p_norm']=='L2':
               success, adv_images, perturbation= penalty_attack(model, images, labels,penalty_type=item['penalty_type'],NumClass=item['NumClass'],mu=item['mu'],alpha=item['alpha'],out_loop_num=item['out_loop_num'],inner_iter_max=item['inner_iter_max'],StepSize=item['StepSize'],p_norm=item['p_norm'],targeted_labels=item['targeted_labels'],beta=item['beta'],rho=item['rho'],loss_type=item['loss_type'],Use_RMS=item['Use_RMS'])
            else:
               success, adv_images, perturbation= penalty_attack(model, images, labels,penalty_type=item['penalty_type'],NumClass=item['NumClass'],mu=item['mu'],alpha=item['alpha'],out_loop_num=item['out_loop_num'],inner_iter_max=item['inner_iter_max'],StepSize=item['StepSize'],p_norm=item['p_norm'],targeted_labels=item['targeted_labels'],beta=item['beta'],rho=item['rho'],loss_type=item['loss_type'])
            outputs = model(adv_images)
            _, labels_predict = torch.max(outputs, 1)
            for i in range(item['batch_size']):
                # define subplot
                plt.subplot(7,item['batch_size'], cnt*item['batch_size']+1 + i)
                #plt.subplot(1,10,1 + i)
                # plot raw pixel data
                plt.axis('off')
                plt.imshow(np.transpose(torchvision.utils.make_grid(adv_images[i].cpu().data, normalize=True),(1,2,0)))
                plt.title('Predict:{}'.format(labels_predict[i].item()), fontsize='xx-small', x=0.5, y=0.9)
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
            # show the figure
            plt.show()
        #    plt.subplots_adjust(wspace=0.1, hspace=0.1)
            cnt+=1

