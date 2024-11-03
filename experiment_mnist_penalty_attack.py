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
from adv_lib.utils.attack_utils import run_attack
from adv_lib.attacks import alma, ddn,fmn,fab,fast_minimum_norm
from functools import partial
from adv_lib.utils.lagrangian_penalties import all_penalties
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
{'model':'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model': 'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model': 'LeNet5','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},

{'model': 'LeNet5','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.99,'loss_type':'CW'},
{'model': 'LeNet5','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.99,'loss_type':'CW'},
{'model': 'LeNet5','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.99,'loss_type':'CW'},

{'model':'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':0.1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},

{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},
{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},
{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},

{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},

{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'max','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},
{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},
{'model': 'ddn','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':10,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':300,'StepSize':0.001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW'},

{'model': 'LeNet5','attack':'DDNL2','batch_size':batch_size ,'iter_num':1000},
{'model': 'Standard','attack':'DDNL2','batch_size':batch_size ,'iter_num':1000},
{'model': 'ddn','attack':'DDNL2','batch_size':batch_size ,'iter_num':1000},

{'model': 'LeNet5','attack':'EADL1','batch_size':batch_size ,'iter_num':1000},
{'model': 'Standard','attack':'EADL1','batch_size':batch_size ,'iter_num':1000},
{'model': 'ddn','attack':'EADL1','batch_size':batch_size ,'iter_num':1000},

{'model':'LeNet5','attack':'CWL2','iter_max':10000,'lr': 0.01,'batch_size':batch_size,'c':10},
{'model':'Standard','attack':'CWL2','iter_max':10000,'lr': 0.01,'batch_size':batch_size,'c':10},
{'model':'ddn','attack':'CWL2','iter_max':10000,'lr': 0.01,'batch_size':batch_size,'c':10},

{'model':'LeNet5','attack':'FMN','p_norm':1,'steps':1000,'batch_size':batch_size,'γ_init':0.05,'α_init':0.01},
{'model':'Standard','attack':'FMN','p_norm':1,'steps':1000,'batch_size':batch_size,'γ_init':0.05,'α_init':0.1},
{'model':'ddn','attack':'FMN','p_norm':1,'steps':1000,'batch_size':batch_size,'γ_init':0.05,'α_init':0.1},

{'model':'LeNet5','attack':'DeepFoolL2','iter_max':100,'batch_size':batch_size},
{'model':'Standard','attack':'DeepFoolL2','iter_max':100,'batch_size':batch_size},
{'model':'ddn','attack':'DeepFoolL2','iter_max':100,'batch_size':batch_size},

# fab in torch attack,but FAB_L1 consistently fails
{'model':'LeNet5','attack':'FAB','p_norm':'L2','steps':200,'eps':100.0,'batch_size':batch_size},
{'model':'Standard','attack':'FAB','p_norm':'L2','steps':200,'eps':100.0,'batch_size':batch_size},
{'model':'ddn','attack':'FAB','p_norm':'L2','steps':200,'eps':100.0,'batch_size':batch_size},

#fab in adversarial library of author of ALMA
{'model':'LeNet5','attack':'FAB_advlib','p_norm':1,'steps':1000,'batch_size':batch_size},
{'model':'Standard','attack':'FAB_advlib','p_norm':1,'steps':1000,'batch_size':batch_size},
{'model':'ddn','attack':'FAB_advlib','p_norm':1,'steps':1000,'batch_size':batch_size},

{'model': 'LeNet5','attack':'ALMA','p_norm':'l2','init_lr_dist':0.1,'batch_size':batch_size ,'iter_num':1000},
{'model': 'Standard','attack':'ALMA','p_norm':'l2','init_lr_dist':0.1,'batch_size':batch_size ,'iter_num':1000},
{'model': 'ddn','attack':'ALMA','p_norm':'l2','init_lr_dist':0.1,'batch_size':batch_size ,'iter_num':1000},


{'model': 'LeNet5','attack':'ALMA','p_norm':'l1','init_lr_dist':0.5,'batch_size':batch_size ,'iter_num':1000},
{'model': 'Standard','attack':'ALMA','p_norm':'l1','init_lr_dist':0.5,'batch_size':batch_size ,'iter_num':1000},
{'model': 'ddn','attack':'ALMA','p_norm':'l1','init_lr_dist':0.5,'batch_size':batch_size ,'iter_num':1000},

    ]

for item in list_para:
    list_success_fail=[]
    list_pert=[]
    list_iterNum=[]
    print(item)
    if item['model']=='LeNet5':
        model = Model_LeNet5()
        model.load_state_dict(torch.load('./MNIST_LeNet_onlyweight.pth'), False)
    else:
        model = model_MNIST()
        if item['model'] == 'Standard':
            model.load_state_dict(torch.load('./models/mnist/mnist_regular.pth'), False)
        elif item['model'] == 'ddn':
            model.load_state_dict(torch.load('./models/mnist/mnist_robust_ddn.pth'), False)  # ALM paper github : l2 adversarially trained
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
            list_pert = list_pert + torch.squeeze(perturbation, dim=1).tolist()
            print('perturbation is: ', torch.t(perturbation))
            print('avg_pert is: ', perturbation.sum() / len(perturbation))
        end_time = time.time()
    elif item['attack']=='ALMA':
        penalty = all_penalties['P2']
        method=partial(alma, penalty=penalty, distance=item['p_norm'], init_lr_distance=item['init_lr_dist'], num_steps=item['iter_num'])
        attack_data = run_attack(model=model, inputs=test_data_normalized[correct_index].cpu(), labels=test_labels[correct_index].cpu(), attack=method, batch_size=item['batch_size'])
        end_time = time.time()
        outs=model(attack_data['adv_inputs'].to(device))
        _,labels_predict=torch.max(outs,dim=1)
        list_success_fail=~torch.eq(labels_predict, test_labels[correct_index])
        if item['p_norm'] == 'l1':
            list_pert = torch.norm((test_data_normalized[correct_index] - attack_data['adv_inputs'].to(device)).view(
                len(test_data_normalized[correct_index]), -1), p=1, dim=1)
        elif item['p_norm'] == 'l2':
            list_pert = torch.norm((test_data_normalized[correct_index] - attack_data['adv_inputs'].to(device)).view(
                len(test_data_normalized[correct_index]), -1), p=2, dim=1)
    elif item['attack']=='DDNL2':
        method=partial(ddn, steps=item['iter_num'])
        attack_data = run_attack(model=model, inputs=test_data_normalized[correct_index].cpu(), labels=test_labels[correct_index].cpu(), attack=method, batch_size=item['batch_size'])
        end_time = time.time()
        outs=model(attack_data['adv_inputs'].to(device))
        _,labels_predict=torch.max(outs,dim=1)
        list_success_fail=~torch.eq(labels_predict, test_labels[correct_index])
        list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]),-1),p=2,dim=1)
    elif item['attack'] == 'FMN':
        method=partial(fmn, norm=item['p_norm'], steps=item['steps'],γ_init=item['γ_init'],α_init=item['α_init'])
        attack_data = run_attack(model=model, inputs=test_data_normalized[correct_index].cpu(), labels=test_labels[correct_index].cpu(), attack=method, batch_size=item['batch_size'])
        end_time = time.time()
        outs=model(attack_data['adv_inputs'].to(device))
        _,labels_predict=torch.max(outs,dim=1)
        list_success_fail=~torch.eq(labels_predict, test_labels[correct_index])
        if item['p_norm']==0:
           list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]),-1),p=0,dim=1)
        elif item['p_norm']==1:
           list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]),-1),p=1,dim=1)
        elif item['p_norm']==2:
           list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]),-1),p=2,dim=1)
        elif item['p_norm']==float('inf'):
           list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]),-1),p=float('inf'),dim=1)
    elif item['attack'] == 'FAB_advlib':
        method=partial(fab, norm=item['p_norm'], n_iter=item['steps'])
        attack_data = run_attack(model=model, inputs=test_data_normalized[correct_index].cpu(), labels=test_labels[correct_index].cpu(), attack=method, batch_size=item['batch_size'])
        end_time = time.time()
        outs=model(attack_data['adv_inputs'].to(device))
        _,labels_predict=torch.max(outs,dim=1)
        list_success_fail=~torch.eq(labels_predict, test_labels[correct_index])
        if item['p_norm']==1:
           list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]),-1),p=1,dim=1)
        elif item['p_norm']==2:
           list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]),-1),p=2,dim=1)
        elif item['p_norm']==float('inf'):
           list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs'].to(device)).view(len(test_data_normalized[correct_index]),-1),p=float('inf'),dim=1)

    elif item['attack']=='EADL1':
        list_const=[]
        for i in range(correct_sum // item['batch_size']):
            print('***************{}th batch***********'.format(i))
            images = test_data_normalized[correct_index[i * item['batch_size']:i * item['batch_size'] + item['batch_size']]].clone().detach()
            labels = test_labels[correct_index[i * item['batch_size']:i * item['batch_size'] + item['batch_size']]]
            atk = torchattacks.EADL1(model, max_iterations=item['iter_num'])  # torchattack
            # adv_image = atk(image, label)
            adv_images, const, _ = atk(images, labels)
            predict_labels=torch.max(model(adv_images),1)[1]
            success_fail=~torch.eq(predict_labels,labels)
            list_success_fail=list_success_fail+success_fail.tolist()
            perturbation=torch.norm((images - adv_images).view(len(images), -1), p=1, dim=1)
            list_pert=list_pert+perturbation.tolist()
            list_const=list_const+const.tolist()
            print('perturbation is: ',torch.t(perturbation))
            print('average of perturbation is:', perturbation.sum() / len(perturbation))
        # The last batch
        if correct_sum % item['batch_size'] != 0:
            i = i + 1
            print('***************{}th batch***********'.format(i))
            images = test_data_normalized[correct_index[i * item['batch_size']:]].clone().detach()
            labels = test_labels[correct_index[i * item['batch_size']:]]
            atk = torchattacks.EADL1(model, max_iterations=item['iter_num'])  # torchattack
            # adv_image = atk(image, label)
            adv_images, const, _ = atk(images, labels)
            predict_labels = torch.max(model(adv_images),1)[1]
            success_fail=~torch.eq(predict_labels,labels)
            list_success_fail=list_success_fail+success_fail.tolist()
            perturbation=torch.norm((images - adv_images).view(len(images), -1), p=1, dim=1)
            list_pert=list_pert+perturbation.tolist()
            list_const = list_const + const.tolist()
            print('perturbation is: ',torch.t(perturbation))
            print('average of perturbation is:', perturbation.sum() / len(perturbation))
        end_time = time.time()
    elif item['attack']=='DeepFoolL2':
        for i in range(correct_sum // item['batch_size']):
            print('***************{}th batch***********'.format(i))
            images = test_data_normalized[correct_index[i * item['batch_size']:i * item['batch_size'] + item['batch_size']]].clone().detach()
            labels = test_labels[correct_index[i * item['batch_size']:i * item['batch_size'] + item['batch_size']]]
            atk = torchattacks.DeepFool(model, steps=item['iter_max'])  # torchattack
            # adv_image = atk(image, label)
            adv_images, num, predict_labels = atk(images, labels)
            success_fail=~torch.eq(predict_labels,labels)
            list_success_fail=list_success_fail+success_fail.tolist()
            perturbation=torch.norm((images - adv_images).view(len(images), -1), p=2, dim=1)
            list_pert=list_pert+perturbation.tolist()
            print('perturbation is: ',torch.t(perturbation))
            print('average of perturbation is:', perturbation.sum() / len(perturbation))
        # The last batch
        if correct_sum % item['batch_size'] != 0:
            i = i + 1
            print('***************{}th batch***********'.format(i))
            images = test_data_normalized[correct_index[i * item['batch_size']:]].clone().detach()
            labels = test_labels[correct_index[i * item['batch_size']:]]
            atk = torchattacks.DeepFool(model, steps=item['iter_max'])  # torchattack
            # adv_image = atk(image, label)
            adv_images, num, predict_labels = atk(images, labels)
            success_fail=~torch.eq(predict_labels,labels)
            list_success_fail=list_success_fail+success_fail.tolist()
            perturbation=torch.norm((images - adv_images).view(len(images), -1), p=2, dim=1)
            list_pert=list_pert+perturbation.tolist()
            print('perturbation is: ',torch.t(perturbation))
            print('average of perturbation is:', perturbation.sum() / len(perturbation))
        end_time = time.time()
    elif item['attack']=='FAB':
        for i in range(correct_sum // item['batch_size']):
            print('***************{}th batch***********'.format(i))
            images = test_data_normalized[correct_index[i * item['batch_size']:i * item['batch_size'] + item['batch_size']]].clone().detach()
            labels = test_labels[correct_index[i * item['batch_size']:i * item['batch_size'] + item['batch_size']]]
            atk = torchattacks.FAB(model, norm=item['p_norm'], steps=item['steps'], eps=item['eps'], n_classes=10)  # torchattack
            adv_images, _, _ = atk(images, labels)
            outs=model(adv_images)
            _, predict_labels=torch.max(outs,dim=1)
            success_fail=~torch.eq(predict_labels,labels)
            list_success_fail=list_success_fail+success_fail.tolist()
            if item['p_norm']=='L2':
               perturbation=torch.norm((images - adv_images).view(len(images), -1), p=2, dim=1)
            elif item['p_norm']=='Linf':
               perturbation=torch.norm((images - adv_images).view(len(images), -1), p=float('inf'), dim=1)
            list_pert=list_pert+perturbation.tolist()
            print('perturbation is: ',torch.t(perturbation))
            print('average of perturbation is:', perturbation.sum() / len(perturbation))
        # The last batch
        if correct_sum % item['batch_size'] != 0:
            i = i + 1
            print('***************{}th batch***********'.format(i))
            images = test_data_normalized[correct_index[i * item['batch_size']:]].clone().detach()
            labels = test_labels[correct_index[i * item['batch_size']:]]
            atk = torchattacks.FAB(model, norm=item['p_norm'], steps=item['steps'], eps=item['eps'], n_classes=10)  # torchattack
            # adv_image = atk(image, label)
            adv_images, _, _ = atk(images, labels)
            outs=model(adv_images)
            _, predict_labels=torch.max(outs,dim=1)
            success_fail=~torch.eq(predict_labels,labels)
            list_success_fail=list_success_fail+success_fail.tolist()
            if item['p_norm']=='L2':
               perturbation=torch.norm((images - adv_images).view(len(images), -1), p=2, dim=1)
            elif item['p_norm']=='Linf':
               perturbation=torch.norm((images - adv_images).view(len(images), -1), p=float('inf'), dim=1)
            list_pert=list_pert+perturbation.tolist()
            print('perturbation is: ',torch.t(perturbation))
            print('average of perturbation is:', perturbation.sum() / len(perturbation))

        end_time = time.time()
    elif item['attack']=='CWL2':
        for i in range(correct_sum // item['batch_size']):
            print('***************{}th batch***********'.format(i))
            images = test_data_normalized[correct_index[i * item['batch_size']:i * item['batch_size'] + item['batch_size']]].clone().detach()
            labels = test_labels[correct_index[i * item['batch_size']:i * item['batch_size'] + item['batch_size']]]
            atk = torchattacks.CW(model, steps=item['iter_max'], lr=item['lr'],c=item['c'])  # torchattack
            # adv_image = atk(image, label)
            adv_images, c, predict_labels = atk(images, labels)
            success_fail=~torch.eq(predict_labels,labels)
            list_success_fail=list_success_fail+success_fail.tolist()
            perturbation=torch.norm((images - adv_images).view(len(images), -1), p=2, dim=1)
            list_pert=list_pert+perturbation.tolist()
            print('perturbation is: ',torch.t(perturbation))
        # The last batch
        if correct_sum % item['batch_size'] != 0:
            i = i + 1
            print('***************{}th batch***********'.format(i))
            images = test_data_normalized[correct_index[i * item['batch_size']:]].clone().detach()
            labels = test_labels[correct_index[i * item['batch_size']:]]
            atk = torchattacks.CW(model, steps=item['iter_max'], lr=item['lr'],c=item['c'])  # torchattack
            # adv_image = atk(image, label)
            adv_images, num, predict_labels = atk(images, labels)
            success_fail=~torch.eq(predict_labels,labels)
            list_success_fail=list_success_fail+success_fail.tolist()
            perturbation=torch.norm((images - adv_images).view(len(images), -1), p=2, dim=1)
            list_pert=list_pert+perturbation.tolist()
            print('perturbation is: ',torch.t(perturbation))
        end_time = time.time()
    time_used=end_time-start_time
    print('running time:',end_time-start_time,'seconds')
    attack_success_rate=sum(list_success_fail)/correct_sum
    print('attack success rate:',attack_success_rate)
    list_pert_success=[ pert  for pert,result in zip(list_pert,list_success_fail) if result]
    avg_pert=sum(list_pert_success)/len(list_pert_success)
    print('avg_pert is',avg_pert)
    dict_save={'device':device,'para':item,'time_used':time_used,'list_success_fail':list_success_fail,'attack_success_rate':attack_success_rate,'list_pert':list_pert,'avg_pert':avg_pert}
    if 'PenaltyAttack' in item['attack']:
        if item['p_norm'] == 'L1':
           torch.save(dict_save,'./result/mnist/{}_attack_{}_pnorm{}_mu{}_alpha{}_outloopnum_{}_inneritermax{}_stepsize{}_penaltytype{}_beta{}_rho{}.pt'.format(item['model'],item['attack'],item['p_norm'],item['mu'],item['alpha'],item['out_loop_num'],item['inner_iter_max'],item['StepSize'],item['penalty_type'],item['beta'],item['rho']))
        else:
           torch.save(dict_save,'./result/mnist/{}_attack_{}_pnorm{}_mu{}_alpha{}_outloopnum_{}_inneritermax{}_stepsize{}_penaltytype{}_beta{}_rho{}.pt'.format(item['model'],item['attack'],item['p_norm'],item['mu'],item['alpha'],item['out_loop_num'],item['inner_iter_max'],item['StepSize'],item['penalty_type'],item['beta'],item['rho']))
    elif 'DDN' in item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_iternum{}.pt'.format(item['model'], item['attack'], item['iter_num']))
    elif 'EADL1' in item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_iternum{}.pt'.format(item['model'], item['attack'], item['iter_num']))
    elif 'DeepFool' in item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_itermax{}.pt'.format(item['model'], item['attack'], item['iter_max']))
    elif 'FAB' == item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_pnorm{}_steps{}.pt'.format(item['model'], item['attack'],item['p_norm'], item['steps']))
    elif 'FAB_advlib'==item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_pnorm{}_steps{}.pt'.format(item['model'], item['attack'],item['p_norm'], item['steps']))
    elif 'CW' in item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_IterMax{}_lr{}.pt'.format(item['model'], item['attack'], item['iter_max'],item['lr']))
    elif 'FMN' in item['attack']:
        torch.save(dict_save, './result/mnist/{}_attack_{}_pnorm{}_steps{}_gammaini{}.pt'.format(item['model'], item['attack'],item['p_norm'],item['steps'],item['γ_init']))
    elif 'ALMA' in item['attack']:
        torch.save(dict_save,'./result/mnist/{}_attack_{}_pnorm{}_iternum{}_initlr{}.pt'.format(item['model'], item['attack'],item['p_norm'], item['iter_num'],item['init_lr_dist']))
