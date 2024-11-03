import os
import torch
import numpy as np
import torchvision
import time
import matplotlib.pyplot as plt
import torchattacks
from adv_lib.utils.attack_utils import run_attack
from adv_lib.attacks import alma, ddn, fmn, fab
from functools import partial
from adv_lib.utils.lagrangian_penalties import all_penalties
import random
from penalty_attack import penalty_attack
from robustbench import  load_model

import gc # garbage collector
def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
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
print(device)
batch_size=64
list_para=[
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':1000,'mu':0.01,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.01,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':1000,'mu':0.01,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.01,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':1000,'mu':0.01,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.01,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#
#{'model': 'WongLinf','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':1000,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'WongLinf','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':1000,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'WongLinf','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':1000,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#
#{'model': 'Salman2020','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'max','NumClass':1000,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'Salman2020','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':1000,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#{'model': 'Salman2020','attack':'PenaltyAttack','p_norm':'L2','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':1000,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.01,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Use_RMS':False},
#
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'max','NumClass':1000,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.0001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Optimizer':'FISTA'},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':1000,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.0001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Optimizer':'FISTA'},
#{'model': 'Standard','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':1000,'mu':1,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.0001,'targeted_labels':None,'beta':0, 'rho':0.999,'loss_type':'CW','Optimizer':'FISTA'},
#
#{'model': 'WongLinf','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'max','NumClass':1000,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW'},
#{'model': 'WongLinf','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':1000,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW'},
#{'model': 'WongLinf','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':1000,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW'},
#
#{'model': 'Salman2020','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'max','NumClass':1000,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW'},
#{'model': 'Salman2020','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'maxsquare','NumClass':1000,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW'},
#{'model': 'Salman2020','attack':'PenaltyAttack','p_norm':'L1','batch_size':batch_size, 'penalty_type':'quadratic','NumClass':1000,'mu':10,'alpha':10.0,'out_loop_num':3,'inner_iter_max':150,'StepSize':0.001,'targeted_labels':None,'beta':0.3, 'rho':0.999,'loss_type':'CW'},
#
#{'model': 'Standard','attack':'DDNL2','batch_size':batch_size ,'iter_num':1000},
#{'model': 'WongLinf','attack':'DDNL2','batch_size':batch_size ,'iter_num':1000},
#{'model': 'Salman2020','attack':'DDNL2','batch_size':batch_size ,'iter_num':1000},
#
#{'model': 'Standard','attack':'EADL1','batch_size':32 ,'iter_num':1000},
#{'model': 'WongLinf','attack':'EADL1','batch_size':32 ,'iter_num':1000},
#{'model': 'Salman2020','attack':'EADL1','batch_size':32 ,'iter_num':1000},
#
#{'model':'Standard','attack':'DeepFoolL2','iter_max':100,'batch_size':batch_size},
#{'model':'WongLinf','attack':'DeepFoolL2','iter_max':100,'batch_size':batch_size},
#{'model':'Salman2020','attack':'DeepFoolL2','iter_max':100,'batch_size':batch_size},
#
#{'model':'WongLinf','attack':'FMN','p_norm':1,'steps':1000,'batch_size':batch_size,'γ_init':0.3,'α_init':0.1},
#{'model':'Standard','attack':'FMN','p_norm':1,'steps':1000,'batch_size':batch_size,'γ_init':0.3,'α_init':0.1},
#{'model':'Salman2020','attack':'FMN','p_norm':1,'steps':1000,'batch_size':batch_size,'γ_init':0.3,'α_init':0.1},
#
#{'model':'WongLinf','attack':'FAB_advlib','p_norm':1,'steps':50,'batch_size':32},#out of memory if batch_size=64
#{'model':'Standard','attack':'FAB_advlib','p_norm':1,'steps':50,'batch_size':32},#out of memory if batch_size=64
#{'model':'Salman2020','attack':'FAB_advlib','p_norm':1,'steps':50,'batch_size':32},#out of memory if batch_size=64

#{'model': 'Standard','attack':'ALMA','p_norm':'l2','init_lr_dist':0.1,'batch_size':batch_size ,'iter_num':1000},
{'model': 'WongLinf','attack':'ALMA','p_norm':'l2','init_lr_dist':0.1,'batch_size':batch_size ,'iter_num':1000},
{'model': 'Salman2020','attack':'ALMA','p_norm':'l2','init_lr_dist':0.1,'batch_size':batch_size ,'iter_num':1000},

{'model': 'Standard', 'attack': 'ALMA', 'p_norm': 'l1', 'init_lr_dist':0.5,'batch_size': batch_size, 'iter_num': 1000},
{'model': 'WongLinf','attack':'ALMA','p_norm':'l1','init_lr_dist':0.5,'batch_size':batch_size ,'iter_num':1000},
{'model': 'Salman2020','attack':'ALMA','p_norm':'l1','init_lr_dist':0.5,'batch_size':batch_size ,'iter_num':1000},
    ]

for item in list_para:
    list_success_fail=[]
    list_pert=[]
    list_iterNum=[]
    print(item)
    images_labels = torch.load('imagenet_first_1000.pt')
    test_data_normalized = images_labels['imgs']
    test_labels = torch.tensor(images_labels['labels']).to(device)
    if item['model'] == 'Standard':
        model = load_model(model_name='Standard_R50',dataset='imagenet', norm='Linf')
    elif item['model'] == 'WongLinf':
        model = load_model(model_name='Wong2020Fast',dataset='imagenet', norm='Linf')
    elif item['model'] == 'Salman2020':
        model = load_model(model_name='Salman2020Do_50_2', dataset='imagenet', norm='Linf')
    model.to(device)
    model.eval()  # turn off the dropout
    test_accuracy = False
    if test_accuracy == True:
        predict_result = torch.tensor([], device=device)
        for i in range(50):
            outputs = model(test_data_normalized[20 * i:20 * i + 20].to(device))
            _, labels_predict = torch.max(outputs, 1)
            predict_result = torch.cat((predict_result, labels_predict), dim=0)
        correct = torch.eq(predict_result, test_labels)
        #imshow(torchvision.utils.make_grid(images_test[0].cpu().data, normalize=True),'Predict:{}'.format(predict_result[0]))
        torch.save(correct, './result/imagenet-first1000/{}_Imagenet_correct_predict.pt'.format(item['model']))
    else:
        correct = torch.load('./result/imagenet-first1000/{}_Imagenet_correct_predict.pt'.format(item['model']))
    correct_sum = correct.sum()
    clean_accuracy = correct_sum / 1000.0
    print('model clean accuracy:', clean_accuracy)
    correct_index=[]
    for i in range(1000):
        if correct[i]:
            correct_index.append(i)
    start_time = time.time()
    if item['attack'] == 'PenaltyAttack':
        for i in range(correct_sum//item['batch_size']):
            print('***************{}th batch***********'.format(i))
            images=test_data_normalized[correct_index[i*item['batch_size']:i*item['batch_size']+item['batch_size']]].to(device)
            labels=test_labels[correct_index[i*item['batch_size']:i*item['batch_size']+item['batch_size']]]

            outputs = model(images)
            _, labels_predict = torch.max(outputs, 1)
            fig = plt.figure(figsize = (5, 15))
            plt.axis('off')
            plt.imshow(np.transpose(torchvision.utils.make_grid(images[1].cpu().data, normalize=True), (1, 2, 0)))
            plt.title('Predict:{}'.format(labels_predict[1].item()), fontsize='xx-small', x=0.5, y=0.9)
            plt.show()
            if item['p_norm']=='L2':
               success, adv_images, perturbation= penalty_attack(model, images, labels,penalty_type=item['penalty_type'],NumClass=item['NumClass'],mu=item['mu'],alpha=item['alpha'],out_loop_num=item['out_loop_num'],inner_iter_max=item['inner_iter_max'],StepSize=item['StepSize'],p_norm=item['p_norm'],targeted_labels=item['targeted_labels'],beta=item['beta'],rho=item['rho'],loss_type=item['loss_type'],Use_RMS=item['Use_RMS'])
            else:
               success, adv_images, perturbation= penalty_attack(model, images, labels,penalty_type=item['penalty_type'],NumClass=item['NumClass'],mu=item['mu'],alpha=item['alpha'],out_loop_num=item['out_loop_num'],inner_iter_max=item['inner_iter_max'],StepSize=item['StepSize'],p_norm=item['p_norm'],targeted_labels=item['targeted_labels'],beta=item['beta'],rho=item['rho'],loss_type=item['loss_type'])
            del adv_images
            torch.cuda.empty_cache()
    #        memReport()
            list_success_fail=list_success_fail+torch.squeeze(success,dim=1).tolist()
            list_pert=list_pert+torch.squeeze(perturbation,dim=1).tolist()
            print('perturbation is: ',torch.t(perturbation))
            print('avg_pert is: ', perturbation.sum() / len(perturbation))
        #The last batch
        if correct_sum%item['batch_size']!=0:
            i=i+1
            print('***************{}th batch***********'.format(i))
            images = test_data_normalized[correct_index[i * item['batch_size']:]].to(device)
            labels = test_labels[correct_index[i * item['batch_size']:]]
            if item['p_norm']=='L2':
               success, adv_images, perturbation= penalty_attack(model, images, labels,penalty_type=item['penalty_type'],NumClass=item['NumClass'],mu=item['mu'],alpha=item['alpha'],out_loop_num=item['out_loop_num'],inner_iter_max=item['inner_iter_max'],StepSize=item['StepSize'],p_norm=item['p_norm'],targeted_labels=item['targeted_labels'],beta=item['beta'],rho=item['rho'],loss_type=item['loss_type'],Use_RMS=item['Use_RMS'])
            else:
               success, adv_images, perturbation= penalty_attack(model, images, labels,penalty_type=item['penalty_type'],NumClass=item['NumClass'],mu=item['mu'],alpha=item['alpha'],out_loop_num=item['out_loop_num'],inner_iter_max=item['inner_iter_max'],StepSize=item['StepSize'],p_norm=item['p_norm'],targeted_labels=item['targeted_labels'],beta=item['beta'],rho=item['rho'],loss_type=item['loss_type'])
            del adv_images
            torch.cuda.empty_cache()
            list_success_fail=list_success_fail+torch.squeeze(success,dim=1).tolist()
            list_pert=list_pert+torch.squeeze(perturbation,dim=1).tolist()
            print('perturbation is: ', torch.t(perturbation))
            print('avg_pert is: ', perturbation.sum() / len(perturbation))
        end_time = time.time()

    elif item['attack']=='ALMA':
        penalty = all_penalties['P2']
        method=partial(alma, penalty=penalty, distance=item['p_norm'], init_lr_distance=item['init_lr_dist'], num_steps=item['iter_num'])
        attack_data = run_attack(model=model, inputs=test_data_normalized[correct_index].cpu(), labels=test_labels[correct_index].cpu(), attack=method, batch_size=item['batch_size'])
        end_time = time.time()
        labels_predict = torch.tensor([], device=device)
        for i in range(correct_sum//4): #evaluate 16 images once
            outputs = model(attack_data['adv_inputs'][16 * i:16 * i + 16].to(device))
            labels_predict = torch.cat((labels_predict, torch.max(outputs, 1)[1]), dim=0)
        if correct_sum%4!=0:
            i=i+1
            outputs = model(attack_data['adv_inputs'][i*16:].to(device))
            labels_predict = torch.cat((labels_predict, torch.max(outputs, 1)[1]), dim=0)
        list_success_fail=~torch.eq(labels_predict, test_labels[correct_index])
        if item['p_norm'] == 'l1':
            list_pert = torch.norm((test_data_normalized[correct_index] - attack_data['adv_inputs']).view(len(test_data_normalized[correct_index]), -1), p=1, dim=1)
        elif item['p_norm'] == 'l2':
            list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs']).view(len(test_data_normalized[correct_index]),-1),p=2,dim=1)

    elif item['attack']=='DDNL2':
        method=partial(ddn, steps=item['iter_num'])
        attack_data = run_attack(model=model, inputs=test_data_normalized[correct_index].cpu(), labels=test_labels[correct_index].cpu(), attack=method, batch_size=item['batch_size'])
        end_time = time.time()
        labels_predict = torch.tensor([], device=device)
        for i in range(correct_sum//16): #evaluate 16 images once
            outputs = model(attack_data['adv_inputs'][16 * i:16 * i + 16].to(device))
            labels_predict = torch.cat((labels_predict, torch.max(outputs, 1)[1]), dim=0)
        if correct_sum%16!=0:
            i=i+1
            outputs = model(attack_data['adv_inputs'][i*16:].to(device))
            labels_predict = torch.cat((labels_predict, torch.max(outputs, 1)[1]), dim=0)
        list_success_fail=~torch.eq(labels_predict, test_labels[correct_index])
        list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs']).view(len(test_data_normalized[correct_index]),-1),p=2,dim=1)

    elif item['attack'] == 'FMN':
        method=partial(fmn, norm=item['p_norm'], steps=item['steps'],γ_init=item['γ_init'],α_init=item['α_init'])
        attack_data = run_attack(model=model, inputs=test_data_normalized[correct_index].cpu(), labels=test_labels[correct_index].cpu(), attack=method, batch_size=item['batch_size'])
        end_time = time.time()
        labels_predict = torch.tensor([], device=device)
        for i in range(correct_sum//16): #evaluate 16 images once
            outputs = model(attack_data['adv_inputs'][16 * i:16 * i + 16].to(device))
            labels_predict = torch.cat((labels_predict, torch.max(outputs, 1)[1]), dim=0)
        if correct_sum%16!=0:
            i=i+1
            outputs = model(attack_data['adv_inputs'][i*16:].to(device))
            labels_predict = torch.cat((labels_predict, torch.max(outputs, 1)[1]), dim=0)
        list_success_fail=~torch.eq(labels_predict, test_labels[correct_index])
        if item['p_norm']==0:
           list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs']).view(len(test_data_normalized[correct_index]),-1),p=0,dim=1)
        elif item['p_norm']==1:
           list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs']).view(len(test_data_normalized[correct_index]),-1),p=1,dim=1)
        elif item['p_norm']==2:
           list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs']).view(len(test_data_normalized[correct_index]),-1),p=2,dim=1)
        elif item['p_norm']==float('inf'):
           list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs']).view(len(test_data_normalized[correct_index]),-1),p=float('inf'),dim=1)

    elif item['attack'] == 'FAB_advlib':
        method=partial(fab, norm=item['p_norm'], n_iter=item['steps'])
        attack_data = run_attack(model=model, inputs=test_data_normalized[correct_index].cpu(), labels=test_labels[correct_index].cpu(), attack=method, batch_size=item['batch_size'])
        end_time = time.time()
        labels_predict = torch.tensor([], device=device)
        for i in range(correct_sum//16): #evaluate 16 images once
            outputs = model(attack_data['adv_inputs'][16 * i:16 * i + 16].to(device))
            labels_predict = torch.cat((labels_predict, torch.max(outputs, 1)[1]), dim=0)
        if correct_sum%16!=0:
            i=i+1
            outputs = model(attack_data['adv_inputs'][i*16:].to(device))
            labels_predict = torch.cat((labels_predict, torch.max(outputs, 1)[1]), dim=0)
        list_success_fail=~torch.eq(labels_predict, test_labels[correct_index])
        if item['p_norm']==0:
           list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs']).view(len(test_data_normalized[correct_index]),-1),p=0,dim=1)
        elif item['p_norm']==1:
           list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs']).view(len(test_data_normalized[correct_index]),-1),p=1,dim=1)
        elif item['p_norm']==2:
           list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs']).view(len(test_data_normalized[correct_index]),-1),p=2,dim=1)
        elif item['p_norm']==float('inf'):
           list_pert=torch.norm((test_data_normalized[correct_index]-attack_data['adv_inputs']).view(len(test_data_normalized[correct_index]),-1),p=float('inf'),dim=1)


    elif item['attack']=='EADL1':
        for i in range(correct_sum // item['batch_size']):
            print('***************{}th batch***********'.format(i))
            images = test_data_normalized[correct_index[i * item['batch_size']:i * item['batch_size'] + item['batch_size']]].to(device)
            labels = test_labels[correct_index[i * item['batch_size']:i * item['batch_size'] + item['batch_size']]]
            atk = torchattacks.EADL1(model, max_iterations=item['iter_num'])  # torchattack
            # adv_image = atk(image, label)
            adv_images,_,_ = atk(images, labels)
            predict_labels=torch.max(model(adv_images),1)[1]
            success_fail=~torch.eq(predict_labels,labels)
            list_success_fail=list_success_fail+success_fail.tolist()
            perturbation=torch.norm((images - adv_images).view(len(images), -1), p=1, dim=1)
            list_pert=list_pert+perturbation.tolist()
            print('perturbation is: ',torch.t(perturbation))
            print('average of perturbation is:', perturbation.sum() / len(perturbation))
        # The last batch
        if correct_sum % item['batch_size'] != 0:
            i = i + 1
            print('***************{}th batch***********'.format(i))
            images = test_data_normalized[correct_index[i * item['batch_size']:]].to(device)
            labels = test_labels[correct_index[i * item['batch_size']:]]
            atk = torchattacks.EADL1(model, max_iterations=item['iter_num'])  # torchattack
            # adv_image = atk(image, label)
            adv_images, num, predict_labels = atk(images, labels)
            predict_labels = torch.max(model(adv_images),1)[1]
            success_fail=~torch.eq(predict_labels,labels)
            list_success_fail=list_success_fail+success_fail.tolist()
            perturbation=torch.norm((images - adv_images).view(len(images), -1), p=1, dim=1)
            list_pert=list_pert+perturbation.tolist()
            print('perturbation is: ',torch.t(perturbation))
            print('average of perturbation is:', perturbation.sum() / len(perturbation))
        end_time = time.time()

    elif item['attack']=='DeepFoolL2':
        for i in range(correct_sum // item['batch_size']):
            print('***************{}th batch***********'.format(i))
            images = test_data_normalized[correct_index[i * item['batch_size']:i * item['batch_size'] + item['batch_size']]].to(device)
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
            images = test_data_normalized[correct_index[i * item['batch_size']:]].to(device)
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
            images = test_data_normalized[correct_index[i * item['batch_size']:i * item['batch_size'] + item['batch_size']]].to(device)
            labels = test_labels[correct_index[i * item['batch_size']:i * item['batch_size'] + item['batch_size']]]
            atk = torchattacks.FAB(model, norm=item['p_norm'], steps=item['steps'], eps=item['eps'], n_classes=1000)  # torchattack
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
        # The last batch
        if correct_sum % item['batch_size'] != 0:
            i = i + 1
            print('***************{}th batch***********'.format(i))
            images = test_data_normalized[correct_index[i * item['batch_size']:]].to(device)
            labels = test_labels[correct_index[i * item['batch_size']:]]
            atk = torchattacks.FAB(model, norm='L2', steps=item['steps'], eps=item['eps'], n_classes=1000)  # torchattack
            # adv_image = atk(image, label)
            adv_images, _, _ = atk(images, labels)
            outs=model(adv_images)
            _, predict_labels=torch.max(outs,dim=1)
            success_fail=~torch.eq(predict_labels,labels)
            list_success_fail=list_success_fail+success_fail.tolist()
            perturbation=torch.norm((images - adv_images).view(len(images), -1), p=2, dim=1)
            list_pert=list_pert+perturbation.tolist()
            print('perturbation is: ',torch.t(perturbation))
            print('average of perturbation is:', perturbation.sum() / len(perturbation))
        end_time = time.time()

    elif item['attack']=='CWL2':
        for i in range(correct_sum // item['batch_size']):
            print('***************{}th batch***********'.format(i))
            images = test_data_normalized[correct_index[i * item['batch_size']:i * item['batch_size'] + item['batch_size']]].to(device)
            labels = test_labels[correct_index[i * item['batch_size']:i * item['batch_size'] + item['batch_size']]]
            atk = torchattacks.CW(model, steps=item['iter_max'], lr=item['lr'])  # torchattack
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
            images = test_data_normalized[correct_index[i * item['batch_size']:]].to(device)
            labels = test_labels[correct_index[i * item['batch_size']:]]
            atk = torchattacks.CW(model, steps=item['iter_max'], lr=item['lr'])  # torchattack
            # adv_image = atk(image, label)
            adv_images, num, predict_labels = atk(images, labels)
            success_fail=~torch.eq(predict_labels,labels)
            list_success_fail=list_success_fail+success_fail.tolist()
            perturbation=torch.norm((images - adv_images).view(len(images), -1), p=2, dim=1)
            list_pert=list_pert+perturbation.tolist()
            print('perturbation is: ',torch.t(perturbation))
            print('average of perturbation is:', perturbation.sum() / len(perturbation))
        end_time = time.time()

    #imshow(torchvision.utils.make_grid(adv_image.cpu().data, normalize=True), 'Predict:{}'.format(predict_label.item()))
    time_used=end_time-start_time
    print('running time:',end_time-start_time,'seconds')
    attack_success_rate=sum(list_success_fail)/correct_sum
    print('attack success rate:',attack_success_rate)
    list_pert_success=[ pert  for pert,result in zip(list_pert,list_success_fail) if result]
    avg_pert=sum(list_pert_success)/len(list_pert_success)
    print('avg_pert is:',avg_pert)
    dict_save={'device':device,'para':item,'time_used':time_used,'list_success_fail':list_success_fail,'attack_success_rate':attack_success_rate,'list_pert':list_pert,'avg_pert':avg_pert}
    if 'PenaltyAttack' in item['attack']:
        if item['p_norm']=='L1':
            torch.save(dict_save,'./result/imagenet-first1000/{}_attack_{}_pnorm{}_penaltytype{}_mu{}_alpha{}_outloopnum_{}_inneritermax{}_stepsize{}_beta{}_rho{}_FISTA.pt'.format(item['model'],item['attack'],item['p_norm'],item['penalty_type'],item['mu'],item['alpha'],item['out_loop_num'],item['inner_iter_max'],item['StepSize'],item['beta'],item['rho']))
        else:
            torch.save(dict_save,'./result/imagenet-first1000/{}_attack_{}_pnorm{}_penaltytype{}_mu{}_alpha{}_outloopnum_{}_inneritermax{}_stepsize{}_beta{}_rho{}.pt'.format(item['model'],item['attack'],item['p_norm'],item['penalty_type'],item['mu'],item['alpha'],item['out_loop_num'],item['inner_iter_max'],item['StepSize'],item['beta'],item['rho']))
    elif 'DDN' in item['attack']:
        torch.save(dict_save,'./result/imagenet-first1000/{}_attack_{}_iternum{}.pt'.format(item['model'], item['attack'], item['iter_num']))
    elif 'PGD' in item['attack'] or 'MIFGSM' in item['attack']:
       torch.save(dict_save,'./result/imagenet-first1000/{}_attack_{}_epsilon{}.pt'.format(item['model'], item['attack'], item['epsilon']))
    elif 'DeepFool' in item['attack']:
        torch.save(dict_save,'./result/imagenet-first1000/{}_attack_{}_itermax{}.pt'.format(item['model'], item['attack'], item['iter_max']))
    elif item['attack']=='FAB':
        torch.save(dict_save,'./result/imagenet-first1000/{}_attack_{}_pnorm{}_steps{}_eps{}.pt'.format(item['model'], item['attack'],item['p_norm'], item['steps'],item['eps']))
    elif item['attack'] == 'FAB_advlib':
        torch.save(dict_save,'./result/imagenet-first1000/{}_attack_{}_pnorm{}_steps{}.pt'.format(item['model'], item['attack'],item['p_norm'], item['steps']))
    elif 'EADL1' in item['attack']:
        torch.save(dict_save,'./result/imagenet-first1000/{}_attack_{}_iternum{}.pt'.format(item['model'], item['attack'],item['iter_num']))
    elif 'CW' in item['attack']:
        torch.save(dict_save,'./result/imagenet-first1000/{}_attack_{}_IterMax{}_lr{}.pt'.format(item['model'], item['attack'], item['iter_max'],item['lr']))
    elif 'FMN' in item['attack']:
        torch.save(dict_save, './result/imagenet-first1000/{}_attack_{}_pnorm{}_steps{}.pt'.format(item['model'], item['attack'],item['p_norm'],item['steps']))
    elif 'ALMA' in item['attack']:
        torch.save(dict_save,'./result/imagenet-first1000/{}_attack_{}_pnorm{}_iternum{}_initlr{}.pt'.format(item['model'], item['attack'],item['p_norm'], item['iter_num'],item['init_lr_dist']))
















