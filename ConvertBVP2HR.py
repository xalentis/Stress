from biosppy import storage
from biosppy.signals import bvp
import numpy as np
import pandas as pd

def extract(subject):

    # rest phase bvp
    signal, mdata = storage.load_txt('./data/bvp_' + subject + '_T1.csv')
    bvp_rate = bvp.bvp(signal=signal, sampling_rate=64, show=False)
    hr1 = bvp_rate[4]

    # speech phase bvp
    signal, mdata = storage.load_txt('./data/bvp_' + subject + '_T2.csv')
    bvp_rate = bvp.bvp(signal=signal, sampling_rate=64, show=False)
    hr2 = bvp_rate[4]

    # arithmetic phase bvp
    signal, mdata = storage.load_txt('./data/bvp_' + subject + '_T3.csv')
    bvp_rate = bvp.bvp(signal=signal, sampling_rate=64, show=False)
    hr3 = bvp_rate[4]

    hr = np.concatenate((hr1, hr2, hr3), axis=None)

    # rest phase eda
    with open('./data/eda_' + subject + '_T1.csv') as file_name:
        eda1 = np.loadtxt(file_name, delimiter=",")

    # speech phase eda
    with open('./data/eda_' + subject + '_T2.csv') as file_name:
        eda2 = np.loadtxt(file_name, delimiter=",")

    # arithmetic phase eda
    with open('./data/eda_' + subject + '_T3.csv') as file_name:
        eda3 = np.loadtxt(file_name, delimiter=",")

    eda = np.concatenate((eda1, eda2, eda3), axis=None)

    df = pd.DataFrame({'eda':eda, 'metric':0})

    with open('./data/selfReportedAnx_' + subject + '.csv') as file_name:
        metric = np.loadtxt(file_name, delimiter=",")

    metric = metric[0][1]

    df['metric'][len(eda1):] = metric

    # export
    np.savetxt(subject + '_hr.csv', hr, delimiter=',')
    df.to_csv(subject + '_eda.csv', index = False)




extract('s46')
extract('s47')
extract('s48')
extract('s49')
extract('s50')
extract('s51')
extract('s52')
extract('s53')
extract('s54')
extract('s55')
extract('s56')
