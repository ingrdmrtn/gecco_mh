import numpy as np, pandas as pd, matplotlib.pyplot as plt


age = 0

if age == 1:

    ns3_young = pd.read_csv('ns3_young_group_age.csv').drop(columns="Unnamed: 0")
    ns6_young = pd.read_csv('ns6_young_group_age.csv').drop(columns="Unnamed: 0")

    ns3_old = pd.read_csv('ns3_old_group_age.csv').drop(columns="Unnamed: 0")
    ns6_old = pd.read_csv('ns6_old_group_age.csv').drop(columns="Unnamed: 0")
    save_file = 'individual_age_ppcs.png'

else:
    ns3_young = pd.read_csv('ns3_young_individual.csv').drop(columns="Unnamed: 0")
    ns6_young = pd.read_csv('ns6_young_individual.csv').drop(columns="Unnamed: 0")

    ns3_old = pd.read_csv('ns3_old_individual.csv').drop(columns="Unnamed: 0")
    ns6_old = pd.read_csv('ns6_old_individual.csv').drop(columns="Unnamed: 0")
    save_file = 'individual_ppcs.png'


ns3_young_real = pd.read_csv('ns3_young_real.csv').drop(columns="Unnamed: 0")
ns6_young_real = pd.read_csv('ns6_young_real.csv').drop(columns="Unnamed: 0")
ns3_old_real = pd.read_csv('ns3_old_real.csv').drop(columns="Unnamed: 0")
ns6_old_real = pd.read_csv('ns6_old_real.csv').drop(columns="Unnamed: 0")

ns3_young_rlwm = pd.read_csv('ns3_young_rlwm.csv').drop(columns="Unnamed: 0")
ns6_young_rlwm = pd.read_csv('ns6_young_rlwm.csv').drop(columns="Unnamed: 0")
ns3_old_rlwm = pd.read_csv('ns3_old_rlwm.csv').drop(columns="Unnamed: 0")
ns6_old_rlwm = pd.read_csv('ns6_old_rlwm.csv').drop(columns="Unnamed: 0")


figure, axis  = plt.subplots(1,2, figsize = (12,5))
axis[0].plot(np.arange(9), np.mean(ns3_young,axis =0), '--',color='b')
axis[0].plot(np.arange(9), np.mean(ns6_young,axis =0), '--',color='r')
axis[0].plot(np.arange(9), np.mean(ns3_young_real,axis =0),'b')
axis[0].plot(np.arange(9), np.mean(ns6_young_real,axis =0),'r')
axis[0].plot(np.arange(9), np.mean(ns3_young_rlwm, axis = 0), 'b', alpha = 0.3)
axis[0].plot(np.arange(9), np.mean(ns6_young_rlwm, axis = 0), 'r', alpha = 0.3)


#
axis[1].plot(np.arange(9), np.mean(ns3_old,axis =0), '--', color = 'b')
axis[1].plot(np.arange(9), np.mean(ns6_old,axis =0), '--', color = 'r')
axis[1].plot(np.arange(9), np.mean(ns3_old_real,axis =0),'b')
axis[1].plot(np.arange(9), np.mean(ns6_old_real,axis =0),'r')
axis[1].plot(np.arange(9), np.mean(ns3_old_rlwm, axis = 0), 'b', alpha = 0.3)
axis[1].plot(np.arange(9), np.mean(ns6_old_rlwm, axis = 0), 'r', alpha = 0.3)


axis[0].set_xticks(np.arange(9))
axis[0].set_xlabel('stimulus iterations')
axis[0].set_ylabel('p(correct)')
axis[0].set_title('young')
axis[0].set_ylim([0,1])
#
axis[1].set_xticks(np.arange(9))
axis[1].set_xlabel('stimulus iterations')
axis[1].set_ylabel('p(correct)')
axis[1].set_title('old')
axis[1].set_ylim([0,1])
#


figure.savefig(f'{save_file}')
print('stop')
