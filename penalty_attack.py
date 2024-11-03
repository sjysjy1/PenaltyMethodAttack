import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()
def abs_fun_grad(x):
    grad=torch.full_like(x,-1.0)
    grad[x>0]=1
    grad[x==0]=0
    return grad
def ISTA_fun(x,x_ori,lamb):
    idx1=torch.ge(x-x_ori,lamb)
    idx2 = torch.le(x-x_ori, -lamb)
    idx3=torch.ge(x-x_ori, -lamb) & torch.le(x-x_ori, lamb)
    x[idx1] = x[idx1]-lamb[idx1]
    x[idx2] = x[idx2]+lamb[idx2]
    x[idx3] = x_ori[idx3]
    return x
def penalty_attack(model:nn.Module,
                   images:Tensor,
                   labels:Tensor,
                   mu: float = 0.1,
                   alpha: float = 10,
                   NumClass: int = 10,
                   penalty_type: str = 'max',
                   out_loop_num:int =4,
                   inner_iter_max:int=50,
                   StepSize:float=0.01,
                   loss_type:str='CW',
                   beta:float=0.3,
                   Use_RMS:bool=True,
                   rho:float=0.99,
                   p_norm:str='Linf',
                   targeted_labels:Tensor=None,
                   Optimizer:str='FISTA',
):
    '''
    :param model: nerual network model
    :param images:(batch_size,channel_num,width,height)
    :param labels: correct label or targeted label
    :param mu: penalty parameter
    :param alpha: scaling factor of mu
    :param penalty_type: 'max','maxsquare','quadratic'
    :param StepSize: initial step size
    :param beta: decaying parameter of momentum
    :param Use_RMS: applying RMSProp or not
    :param rho: decaying parameter of RMSProp
    :param p_norm: 'L1','L2','Linf'
    :param targeted: targeted attack
    '''
    device=images.device
    images_ori=images
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    momentum = torch.zeros_like(images).to(device)
    dir_img=torch.zeros_like(images).to(device)
    RMS_avg = torch.zeros_like(images).to(device)
    RMS_avg_z=0
    stepsize = StepSize
    # we need a small step to avoid being stuck in xk where the gradient is zero.
    images.requires_grad_()
    model.zero_grad()
    outs = model(images)
    outs.gather(1, labels.view(-1, 1)).sum().backward()
    images = torch.clamp(images - stepsize * images.grad, min=0, max=1).detach()
    norm=torch.full((len(images),1),float('inf'),device=device)
    best_adv=images.clone().detach().to(device)
    best_adv_norm=torch.full((len(images),1),float('inf'),device=device)
    for i in range(out_loop_num):
       print('penalty parameter mu is: {}'.format(mu))
       if p_norm=='L1' and Optimizer=='FISTA':
          t = 1
          Y_FISTA = images
          images_last = images
       for cnt in range(inner_iter_max):
          images.requires_grad_()  #################### require gradient#########
          model.zero_grad()  #################### zero gradient#########   why is model??
          outs = model(images)
          if targeted_labels==None:
              one_hot_bool=(F.one_hot(labels,num_classes=NumClass)>0)
              outs_cp=outs.detach().clone()
              outs_cp[one_hot_bool]=-torch.inf
              #outs_cp.gather(1, labels.view(-1, 1))=-torch.inf
              _,MaxIndex_except_label=torch.max(outs_cp,dim=1)
              if loss_type=='DLR':
                  _,SortedIndex=outs.sort(dim=1)
                  outs_diff=(-outs.gather(1, MaxIndex_except_label.view(-1, 1)) + outs.gather(1, labels.view(-1, 1)))/(outs.gather(1, SortedIndex[:,-1].view(-1, 1))-outs.gather(1, SortedIndex[:,-3].view(-1, 1))+1e-12)
              elif loss_type=='CW':
                  outs_diff = -outs.gather(1, MaxIndex_except_label.view(-1, 1)) + outs.gather(1, labels.view(-1, 1))
          else:
              outs_diff = -outs.gather(1, targeted_labels.view(-1, 1)) + outs.gather(1, labels.view(-1, 1))
          outs_diff.sum().backward()
          is_adv =~(outs_diff>0)
          is_better_adv=is_adv & (norm<best_adv_norm)
          is_better_adv_unsqueeze=torch.unsqueeze(torch.unsqueeze(is_better_adv,dim=2),dim=3)
          best_adv_norm = torch.where(is_better_adv, norm, best_adv_norm)
          best_adv = torch.where(is_better_adv_unsqueeze, images.detach(), best_adv)

          #calculating gradient for images and introduced variables z or y
          is_adv = torch.squeeze(is_adv, dim=1)
          if penalty_type=='max' and (p_norm=='L1' or p_norm=='L2'):
             grad_img_penalty_term=mu*images.grad
             grad_img_penalty_term[is_adv] = 0
          elif penalty_type=='max' and (p_norm=='Linf'):
             y = torch.full((len(images), 1), 5, device=device)
             grad_img_penalty_term = images.grad
             grad_img_penalty_term[is_adv] = 0
             grad_img_penalty_term = mu*(grad_img_penalty_term+(torch.sign(images-images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images))+1)/2 - (torch.sign(-images+images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images))+1)/2)
             grad_y=1-mu*(torch.sign(images-images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images))+1).sum()/2 -mu*(torch.sign(-images+images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images))+1).sum()/2

          elif penalty_type=='maxsquare' and (p_norm=='L1' or p_norm=='L2'):
              with torch.no_grad():  #out of memory occur for imagenet if no_grad() is not used
                  grad_img_penalty_term=2*mu*torch.unsqueeze(torch.unsqueeze(outs_diff,dim=2),dim=3).expand_as(images)*images.grad  #to make the penalty differentiable,use: 2*images.grad
                  grad_img_penalty_term[is_adv] = 0
          elif penalty_type=='maxsquare' and (p_norm=='Linf'):
             y = torch.full((len(images), 1), 5, device=device)
             grad_img_penalty_term = 2*mu*torch.unsqueeze(torch.unsqueeze(outs_diff,dim=2),dim=3).expand_as(images)*images.grad
             grad_img_penalty_term[is_adv] = 0
             grad_img_penalty_term = grad_img_penalty_term+2*mu*(images-images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images))*(torch.sign(images-images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images))+1)/2 \
                                                             +2*mu*(images-images_ori+torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images))*(torch.sign(-images+images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images))+1)/2
             grad_y=1-2*mu*((images-images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images))*(torch.sign(images-images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images))+1)).sum()/2 \
                     +2*mu*((images-images_ori+torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images))*(torch.sign(-images+images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images))+1)).sum()/2

          elif penalty_type=='quadratic' and (p_norm=='L1' or p_norm=='L2'):
              with torch.no_grad():#out of memory occur for imagenet if no_grad() is not used
                  z=torch.full((len(images),1),0.1,device=device)
                  #grad_img_penalty_term = 2*(outs_diff+z**2)*images.grad
                  grad_img_penalty_term =mu*( 2 * torch.unsqueeze(torch.unsqueeze(outs_diff + z ** 2,2),3).expand_as(images.grad) * images.grad )
                  grad_z=mu*2*(outs_diff+z**2)*2*z  #in ICME version mu is forgotton????
          elif penalty_type == 'quadratic' and p_norm=='Linf':
              z = torch.full((len(images), 1), 1.0, device=device)
              y=torch.full((len(images),1),10,device=device)
              s = torch.full_like(images, 1.0, device=device)
              t = torch.full_like(images, 1.0, device=device)
              grad_img_penalty_term = 2 * torch.unsqueeze(torch.unsqueeze(outs_diff + z ** 2, 2), 3).expand_as(images.grad) * images.grad
              grad_img_penalty_term = mu*(grad_img_penalty_term+2*(images-images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images)+s**2) -2*(-images+images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images)+t**2))
              grad_z=mu*2*(outs_diff+z**2)*2*z
              grad_y =1-mu*2*(images-images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images)+s**2).sum()-mu*2*(-images+images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images)+t**2).sum()
              grad_s =mu*2*(images-images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images)+s**2)*2*s
              grad_t =mu*2*(-images+images_ori-torch.unsqueeze(torch.unsqueeze(y,dim=2),dim=3).expand_as(images)+t**2)*2*t
          elif penalty_type == 'absolute' and (p_norm == 'L1' or p_norm == 'L2'):
              z = torch.full((len(images), 1), 0.1, device=device)
              grad_img_penalty_term =mu*torch.sign( torch.unsqueeze(torch.unsqueeze(outs_diff + z ** 2,2),3).expand_as(images.grad)) *(images.grad)
              grad_z=mu*2*z*torch.sign(outs_diff + z ** 2)

          #updating image
          if p_norm=='L2':
              with torch.no_grad():
               dir_img_basic=-2*(images-images_ori)-grad_img_penalty_term
               dir_img=beta*dir_img+dir_img_basic
          elif p_norm=='L1':
               #grad_img_obj=torch.sign(images-images_ori) #what is the value of torch.sign(0)?
               grad_img_obj = abs_fun_grad(images - images_ori)
               dir_img_basic=-grad_img_obj-grad_img_penalty_term
               dir_img = beta * dir_img + dir_img_basic

          elif p_norm=='Linf':
               dir_img=-grad_img_penalty_term

          if Use_RMS:
             RMS_avg = rho * RMS_avg + (1 - rho) * dir_img ** 2
             stepsize=StepSize/(RMS_avg**0.5+0.000000001)
          else:
              stepsize =torch.full_like(images,StepSize,device=device)


          if p_norm=='L1' and Optimizer=='ISTA':
              images=(images-stepsize*grad_img_penalty_term)
              images=torch.clamp(ISTA_fun(images,images_ori,stepsize),min=0.0,max=1.0).detach()
          elif p_norm=='L1' and Optimizer=='FISTA':
              t_last=t
              tmp = images
              Y_FISTA = (Y_FISTA - stepsize * grad_img_penalty_term)
              t=(1+(1+4*t*t)**0.5)/2
              images=torch.clamp(ISTA_fun(Y_FISTA,images_ori,stepsize),min=0.0,max=1.0).detach()
              images_last=tmp
              Y_FISTA=images+(t_last-1)/t*(images-images_last)
          else: #using gradient descent
              images=torch.clamp(images+stepsize*dir_img,min=0.0,max=1.0).detach()  #mark:starstar  #if detach() is removed, error will occur below: NoneType

          #updating introduced variables
          if  p_norm=='L1' or p_norm=='L2':
              if penalty_type == 'quadratic' or penalty_type == 'absolute':
                 dir_z = -grad_z
                 RMS_avg_z = rho * RMS_avg_z + (1 - rho) * dir_z ** 2
                 stepsize_z = StepSize / (RMS_avg_z ** 0.5 + 0.000000001)
                 z=z+stepsize_z*dir_z
          elif  p_norm=='Linf':
              y = y - stepsize * grad_y
              if penalty_type == 'quadratic':
                 z=z-stepsize*grad_z
                 s = s - stepsize * grad_s
                 t = t - stepsize * grad_t

          if p_norm=='L2':
              norm=torch.unsqueeze(torch.norm((images-images_ori).view(len(images),-1),p=2,dim=1),dim=1)
          elif p_norm == 'L1':
              norm = torch.unsqueeze(torch.norm((images - images_ori).view(len(images), -1), p=1, dim=1), dim=1)
          elif p_norm == 'Linf':
              norm = torch.unsqueeze(torch.norm((images - images_ori).view(len(images), -1), p=float('inf'), dim=1), dim=1)
       mu=alpha*mu

    if loss_type == 'DLR':
        _, SortedIndex = outs.sort(dim=1)
        outs_diff = (-outs.gather(1, MaxIndex_except_label.view(-1, 1)) + outs.gather(1, labels.view(-1, 1))) / (
                    outs.gather(1, SortedIndex[:, -1].view(-1, 1)) - outs.gather(1, SortedIndex[:, -3].view(-1,1)) + 1e-12)
    elif loss_type == 'CW':
        outs_diff = -outs.gather(1, MaxIndex_except_label.view(-1, 1)) + outs.gather(1, labels.view(-1, 1))
    is_adv =~(outs_diff>0)
    is_better_adv=is_adv & (norm<best_adv_norm)
    is_better_adv_unsqueeze=torch.unsqueeze(torch.unsqueeze(is_better_adv,dim=2),dim=3)
    best_adv_norm = torch.where(is_better_adv, norm, best_adv_norm)
    best_adv = torch.where(is_better_adv_unsqueeze, images.detach(), best_adv)

    success=(best_adv_norm!=torch.inf)
    return success,best_adv,best_adv_norm

