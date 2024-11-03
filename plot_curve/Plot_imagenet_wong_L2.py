import torch
import numpy as np
import matplotlib.pyplot as plt
wong_imagenet_L2_penaltymax=torch.load('../result/imagenet-first1000/WongLinf_attack_PenaltyAttack_pnormL2_penaltytypemax_mu10_alpha10.0_outloopnum_3_inneritermax150_stepsize0.01_beta0_rho0.999.pt')
pert_list1=np.asarray(wong_imagenet_L2_penaltymax['list_pert'])
wong_imagenet_L2_penaltymaxsquare=torch.load('../result/imagenet-first1000/WongLinf_attack_PenaltyAttack_pnormL2_penaltytypemaxsquare_mu10_alpha10.0_outloopnum_3_inneritermax150_stepsize0.01_beta0_rho0.999.pt')
pert_list2=np.asarray(wong_imagenet_L2_penaltymaxsquare['list_pert'])
wong_imagenet_L2_penaltyquadratic=torch.load('../result/imagenet-first1000/WongLinf_attack_PenaltyAttack_pnormL2_penaltytypequadratic_mu10_alpha10.0_outloopnum_3_inneritermax150_stepsize0.01_beta0_rho0.999.pt')
pert_list3=np.asarray(wong_imagenet_L2_penaltyquadratic['list_pert'])

wong_L2attack_DeepFool=torch.load('../result/imagenet-first1000/WongLinf_attack_DeepFoolL2_itermax100.pt')
pert_list5=np.asarray(wong_L2attack_DeepFool['list_pert'])
wong_L2attack_CW=torch.load('../result/imagenet-first1000/WongLinf_attack_CWL2_IterMax10000_lr0.001.pt')
pert_list6=np.asarray(wong_L2attack_CW['list_pert'])
wong_L2attack_DDN=torch.load('../result/imagenet-first1000/WongLinf_attack_DDNL2_iternum1000.pt')
pert_list7=np.asarray(wong_L2attack_DDN['list_pert'].cpu())
wong_L2attack_ALMA=torch.load('../result/imagenet-first1000/WongLinf_attack_ALMA_pnorml2_iternum1000_initlr0.1.pt')
pert_list8=np.asarray(wong_L2attack_ALMA['list_pert'].cpu())



x=np.linspace(0,5,500)
y1=[len(pert_list1[np.asarray(pert_list1<item) & np.asarray(wong_imagenet_L2_penaltymax['list_success_fail'])])/len(pert_list1) for item in x ]
y2=[len(pert_list2[np.asarray(pert_list2<item) & np.asarray(wong_imagenet_L2_penaltymaxsquare['list_success_fail'])])/len(pert_list2) for item in x ]
y3=[len(pert_list3[np.asarray(pert_list3<item) & np.asarray(wong_imagenet_L2_penaltyquadratic['list_success_fail'])])/len(pert_list3) for item in x ]
#FAB is not used
y5=[len(pert_list5[np.asarray(pert_list5<item) & np.asarray(wong_L2attack_DeepFool['list_success_fail'])])/len(pert_list5) for item in x ]
y6=[len(pert_list6[np.asarray(pert_list6<item) & np.asarray(wong_L2attack_CW['list_success_fail'])])/len(pert_list6) for item in x ]
y7=[len(pert_list7[np.asarray(pert_list7<item) & np.asarray(wong_L2attack_DDN['list_success_fail'].cpu())])/len(pert_list7) for item in x ]
y8=[len(pert_list8[np.asarray(pert_list8<item) & np.asarray(wong_L2attack_ALMA['list_success_fail'].cpu())])/len(pert_list8) for item in x ]

fig = plt.figure(figsize=(6,4))
plt.plot(x,y1,linestyle='--', color = 'b', linewidth=1, label='PenaltyAttack-P1')
plt.plot(x,y2,linestyle='--', color = 'r', linewidth=1, label='PenaltyAttack-P2')
plt.plot(x,y3,linestyle='--', color = 'lime', linewidth=1, label='PenaltyAttack-P3')
#FAB is not used
plt.plot(x,y5,linestyle='--', color = 'violet', linewidth=1, label='DeepFool')
plt.plot(x,y6,linestyle='--', color = 'purple', linewidth=1, label='CW')
plt.plot(x,y7,linestyle='--', color = 'silver', linewidth=1, label='DDN')
plt.plot(x,y8,linestyle='--', color = 'springgreen', linewidth=1, label='ALMA')
plt.legend(loc="lower right")
plt.legend(fontsize="8",loc="best")
plt.xlabel('Pert')
plt.ylabel('ASR')
plt.title("L2 attack on WongLinf-I")
plt.grid(True)
print('*******')



