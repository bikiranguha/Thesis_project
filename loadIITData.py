import pandas as pd 
import matplotlib.pyplot as plt
# column_names = ['time', 'SBL_VAB_m', 'SBL_VAB_a', 'SBL_VBC_m', 'SBL_VBC_a', 'SBL_VCA_m', 'SBL_VCA_a', 'SBL_IA_m', 'SBL_IA_a', 'SBL_IB_m', 'SBL_IB_a', 'SBL_IC_m', 'SBL_IC_a',
#                     'SBP_VAB_m', 'SBP_VAB_a', 'SBP_VBC_m', 'SBP_VBC_a', 'SBP_VCA_m', 'SBP_VCA_a', 'SBP_IA_m', 'SBP_IA_a', 'SBP_IB_m', 'SBP_IB_a', 'SBP_IC_m', 'SBP_IC_a']
# iitdata = pd.read_csv('testIIT.csv', names = column_names)
# #iitdata = pd.read_csv('testIIT.csv')

# singleVSample = iitdata[['time','SBL_VAB_m']]
# singleVSample.to_csv('IIT_SBL_VAB.csv',index=False)

IITVmSample = pd.read_csv('IIT_SBL_VAB.csv')
vpu = IITVmSample.SBL_VAB_m
vpu = vpu/vpu.mean()
plt.plot(IITVmSample.time, vpu)
plt.ylim(0,1.2)
plt.grid()
plt.show()
