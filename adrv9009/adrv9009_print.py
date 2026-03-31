


import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import adi

# Create radio
sdr = adi.adrv9009(uri="ip:192.168.1.10")

print(sdr.ensm_mode)
print(sdr.profile)
print(sdr.frequency_hopping_mode)
print(sdr.frequency_hopping_mode_en)
print(sdr.calibrate_rx_phase_correction_en)
print(sdr.calibrate_rx_qec_en)
print(sdr.calibrate_tx_qec_en)
print(sdr.calibrate)
print(sdr.gain_control_mode_chan0)
print(sdr.gain_control_mode_chan1)

print(sdr.obs_rf_port_select)
print(sdr.jesd204_fsm_state)
