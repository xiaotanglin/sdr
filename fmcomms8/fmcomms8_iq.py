import matplotlib.pyplot as plt
import numpy as np
import adi

# from fmcomms8 import fmcomms8

sdr = adi.fmcomms8(uri="ip:192.168.1.10")

#设置
sdr.rx_enabled_channels=[0]
sdr.tx_enabled_channels=[0]

sdr.trx_lo=2_000_000_000
sdr.trx_lo_chip_b = 2000000000
sdr.tx_hardwaregain_chan0 = -10
sdr.tx_hardwaregain_chan1 = -10
sdr.tx_hardwaregain_chan0_chip_b = -10
sdr.tx_hardwaregain_chan1_chip_b = -10
sdr.gain_control_mode_chan0 = "slow_attack"
sdr.gain_control_mode_chan1 = "slow_attack"
sdr.gain_control_mode_chan0_chip_b = "slow_attack"
sdr.gain_control_mode_chan1_chip_b = "slow_attack"
sdr.rx_buffer_size = 2 ** 17

# Read properties
print("TRX LO %s" % (sdr.trx_lo))
print("TRX LO %s" % (sdr.trx_lo_chip_b))

# Send data 20Mhz

sdr.dds_enabled = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
sdr.dds_frequencies = [20000000, 0, 20000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
sdr.dds_scales = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
sdr.dds_phases = [0, 0, 90000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


#数据接受
data=sdr.rx()
sig1=data
# sig2=data[1]

I1=np.real(sig1)
Q1=np.imag(sig1)
#画图 IQ数据
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(I1,color='blue',linewidth=1)
plt.title(f"channel{0}-I signal",fontsize=14)
plt.subplot(2,1,2)
plt.plot(Q1, color='blue', linewidth=1)
plt.title(f"channel {0} - Q signal", fontsize=14)

plt.grid(True)
plt.show()

#fft
Fs=122_880_000
N = len(data)
fft_vals = np.fft.fft(data)
fft_freq = np.fft.fftfreq(N,1/Fs)
fft_vals = np.fft.fftshift(fft_vals)
fft_freq = np.fft.fftshift(fft_freq)


# 计算幅度（dB）
mag = 20 * np.log10(np.abs(fft_vals))

# 5. 自动找最强频点
peak_idx = np.argmax(mag)
peak_freq = fft_freq[peak_idx]
peak_mag = mag[peak_idx]

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