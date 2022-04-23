import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    df_syn = pd.read_csv('../build/synapseInfos.csv')
    df_syn = df_syn[df_syn.IsInhibitory == False]
    df_spike = pd.read_csv('../build/spikeTrains.csv')

    # df_weight = pd.read_csv('../build/weightProgression.csv')
    # df_weight.set_index('Time')
    # df_weight.plot(figsize=(30, 20), x='Time')

    target_time = 0.5
    window = 1.0

    df_spike = df_spike[abs(df_spike.Time - target_time) <= window * 0.5]
    # fig, plts = plt.subplots(2)
    # plts[0].scatter(df_spike.Time, df_spike.NeuronId, s=1)
    # plts[1].hist(df_syn.Weight, 100)
    # plt.show()
    plt.figure(figsize=(10, 5))
    plt.scatter(df_spike.Time, df_spike.NeuronId, s=1, color='black')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Neuron #', fontsize=12)
    plt.show()
    #plt.savefig('SpikeRaster.png')
