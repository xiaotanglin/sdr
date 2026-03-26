
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import adi


# Create radio
sdr = adi.FMComms5(uri="ip:192.168.1.10")

# Configure properties
sdr.rx_lo = 2000000000
sdr.rx_lo_chip_b = 2000000000
sdr.tx_lo = 2000000000
sdr.tx_lo_chip_b = 2000000000
sdr.tx_cyclic_buffer = True
sdr.tx_hardwaregain_chan0 = -30
sdr.tx_hardwaregain_chip_b_chan0 = -30
sdr.gain_control_mode_chan0 = "slow_attack"
sdr.gain_control_mode_chip_b_chan0 = "slow_attack"
sdr.sample_rate = 1000000


# Set single DDS tone for TX on one transmitter
sdr.dds_single_tone(30000, 0.9)


data=sdr.rx()

sig = data[0]

I=np.real(sig)
Q=np.imag(sig)

plt.figure(figsize=(12, 6))

plt.subplot(2,1,1)
plt.plot(I, color='blue', linewidth=1)
plt.title(f"channel {0} - I signal", fontsize=14)
plt.grid(True)
plt.show()

Fs=sdr.sample_rate

fft_vals = np.fft.fft(sig)
fft_freq = np.fft.fftfreq(1024,1/Fs)
fft_vals = np.fft.fftshift(fft_vals)
fft_freq = np.fft.fftshift(fft_freq)

# 计算幅度（dB）
mag = 20 * np.log10(np.abs(fft_vals))

# 5. 自动找最强频点
peak_idx = np.argmax(mag)
peak_freq = fft_freq[peak_idx]
peak_mag = mag[peak_idx]

# print(data[0])
# 6. 画图
plt.figure(figsize=(12, 6))
plt.plot(fft_freq, mag, linewidth=1.5)
plt.grid(True)

# 标出峰值频点
plt.text(peak_freq + 200000, peak_mag - 5, 
         f"Freq: {peak_freq/1e6:.2f} MHz", 
         color="red", fontsize=13)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title(f"FFT Spectrum - Channel 0 | Sample Rate: {Fs/1e6:.2f} MSPS")
plt.tight_layout()
plt.show()