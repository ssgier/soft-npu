import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    df_syn = pd.read_csv('~/git/soft_npu/build/synapseInfos.csv')
    df_syn = df_syn[df_syn.IsInhibitory == False]
    df_spike = pd.read_csv('~/git/soft_npu/build/spikeTrains.csv')

    df_weight = pd.read_csv('~/git/soft_npu/build/weightProgression.csv')
    df_weight.set_index('Time')

    df_weight.plot(figsize=(30, 20), x='Time')

    plt.show()
