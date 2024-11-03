import torch
import numpy as np
import matplotlib.pyplot as plt
WangL2_cifar_L1_penaltymax=torch.load('../result/cifar10-first1000/WangL2WRN-28-10_attack_PenaltyAttack_pnormL1_penaltytypemax_mu10_alpha10.0_outloopnum_3_inneritermax150_stepsize0.001_beta0.3_rho0.999.pt')
pert_list1=np.asarray(WangL2_cifar_L1_penaltymax['list_pert'])
WangL2_cifar_L1_penaltymaxsquare=torch.load('../result/cifar10-first1000/WangL2WRN-28-10_attack_PenaltyAttack_pnormL1_penaltytypemaxsquare_mu10_alpha10.0_outloopnum_3_inneritermax150_stepsize0.001_beta0.3_rho0.999.pt')
pert_list2=np.asarray(WangL2_cifar_L1_penaltymaxsquare['list_pert'])
WangL2_cifar_L1_penaltyquadratic=torch.load('../result/cifar10-first1000/WangL2WRN-28-10_attack_PenaltyAttack_pnormL1_penaltytypequadratic_mu10_alpha10.0_outloopnum_3_inneritermax150_stepsize0.001_beta0.3_rho0.999.pt')
pert_list3=np.asarray(WangL2_cifar_L1_penaltyquadratic['list_pert'])
WangL2_L1attack_FAB=torch.load('../result/cifar10-first1000/WangL2WRN-28-10_attack_FAB_advlib_pnorm1_steps200.pt')
pert_list4=np.asarray(WangL2_L1attack_FAB['list_pert'].cpu())
WangL2_L1attack_EAD=torch.load('../result/cifar10-first1000/WangL2WRN-28-10_attack_EADL1_iternum1000.pt')
pert_list5=np.asarray(WangL2_L1attack_EAD['list_pert'])
WangL2_L1attack_FMN=torch.load('../result/cifar10-first1000/WangL2WRN-28-10_attack_FMN_pnorm1_steps1000.pt')
pert_list6=np.asarray(WangL2_L1attack_FMN['list_pert'].cpu())
WangL2_L1attack_ALMA=torch.load('../result/cifar10-first1000/WangL2WRN-28-10_attack_ALMA_pnorml1_iternum1000_initlr0.5.pt')
pert_list7=np.asarray(WangL2_L1attack_ALMA['list_pert'].cpu())

x=np.linspace(0,50,50000)
y1=[len(pert_list1[np.asarray(pert_list1<item) & np.asarray(WangL2_cifar_L1_penaltymax['list_success_fail'])])/len(pert_list1) for item in x ]
y2=[len(pert_list2[np.asarray(pert_list2<item) & np.asarray(WangL2_cifar_L1_penaltymaxsquare['list_success_fail'])])/len(pert_list2) for item in x ]
y3=[len(pert_list3[np.asarray(pert_list3<item) & np.asarray(WangL2_cifar_L1_penaltyquadratic['list_success_fail'])])/len(pert_list3) for item in x ]
y4=[len(pert_list4[np.asarray(pert_list4<item) & np.asarray(WangL2_L1attack_FAB['list_success_fail'].cpu())])/len(pert_list4) for item in x ]
y5=[len(pert_list5[np.asarray(pert_list5<item) & np.asarray(WangL2_L1attack_EAD['list_success_fail'])])/len(pert_list5) for item in x ]
y6=[len(pert_list6[np.asarray(pert_list6<item) & np.asarray(WangL2_L1attack_FMN['list_success_fail'].cpu())])/len(pert_list6) for item in x ]
y7=[len(pert_list7[np.asarray(pert_list7<item) & np.asarray(WangL2_L1attack_ALMA['list_success_fail'].cpu())])/len(pert_list7) for item in x ]

fig = plt.figure(figsize=(6,4))
plt.plot(x,y1,linestyle='--', color = 'b', linewidth=1, label='PenaltyAttack-P1')
plt.plot(x,y2,linestyle='--', color = 'r', linewidth=1, label='PenaltyAttack-P2')
plt.plot(x,y3,linestyle='--', color = 'lime', linewidth=1, label='PenaltyAttack-P3')
plt.plot(x,y4,linestyle='--', color = 'g', linewidth=1, label='FAB')
plt.plot(x,y5,linestyle='--', color = 'violet', linewidth=1, label='EAD')
plt.plot(x,y6,linestyle='--', color = 'silver', linewidth=1, label='FMN')
plt.plot(x,y7,linestyle='--', color = 'springgreen', linewidth=1, label='ALMA')

plt.legend(loc="best")
plt.xlabel('Pert')
plt.ylabel('ASR')
plt.title("L1 attack on WangL2-C")
plt.grid(True)
print('*******')



