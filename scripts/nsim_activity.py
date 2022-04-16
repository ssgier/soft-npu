import sys, pygame
import time
import pandas as pd
pygame.init()

size = width, height = 633, 633
BLACK = 0, 0, 0
BLUE = 0, 0, 255
RED = 255, 0, 0
WHITE = 255, 255, 255
flash_time = 10e-4
dt = 1e-4
start_time = 38.0

logging_interval = 100e-3

print('reading simulation data')

df_spike = pd.read_csv('~/git/soft_npu/build/spikeTrains.csv')
df_spike = df_spike[df_spike['Time'] >= start_time]
df_locations = pd.read_csv('~/git/soft_npu/build/locations.csv')
df_neuron_infos = pd.read_csv('~/git/soft_npu/build/neuronInfos.csv')

assert len(df_locations) == len(df_neuron_infos)

num_neurons = len(df_neuron_infos)
inhibitory_flags = df_neuron_infos['IsInhibitory'].transform(bool).values.tolist()
screen_locations = df_locations[['LocationX', 'LocationY']].apply(lambda x: (int(x[0] * width), int((1 - x[1]) * height)), axis=1).values.tolist()

screen = pygame.display.set_mode(size)

t = start_time
neuron_id_to_last_spike_t = {}
last_logging_time = 0

print('starting visualization')

for index, row in df_spike.iterrows():
    spike_t = row['Time']

    while t < spike_t:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        screen.fill(WHITE)
        neuron_id_to_last_spike_t_next = {}
        for neuron_id, last_spike_t in neuron_id_to_last_spike_t.items():
            if t - last_spike_t <= flash_time:
                neuron_id_to_last_spike_t_next[neuron_id] = last_spike_t
                color = BLUE if inhibitory_flags[neuron_id] else RED
                pygame.draw.circle(screen, color, screen_locations[neuron_id], 1)
        neuron_id_to_last_spike_t = neuron_id_to_last_spike_t_next
        pygame.display.flip()
        if t - last_logging_time > logging_interval:
            print('Reached time {}'.format(t))
            last_logging_time = t
        t += dt
        time.sleep(1e-4)
    neuron_id_to_last_spike_t[int(row['NeuronId'])] = spike_t
