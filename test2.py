import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import adi

# Create radio
sdr = adi.FMComms5(uri="ip:192.168.1.10")


print(sdr.rx_annotated)

print(sdr.rx_output_type)

print(sdr. rx_enabled_channels)

print(sdr._num_rx_channels_enabled)

print(sdr.rx())


print(sdr._complex_data)

print(sdr._complex_data)

print(sdr._rx_init_channels)

print(sdr.tx_cyclic_buffer)

print(sdr._num_tx_channels_enabled)

print(sdr.tx_enabled_channels)

print(sdr.tx_channel_names)

sdr.tx_enabled_channels=[0]

print(sdr.tx_enabled_channels)

print(sdr.tx_channel_names)

print(sdr._num_tx_channels_enabled)