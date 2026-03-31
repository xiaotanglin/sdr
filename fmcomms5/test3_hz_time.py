#计算时间频率图，横坐标是频率，纵坐标是时间

import adi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# 1. SDR 配置
sdr = adi.FMComms5(uri="ip:192.168.1.10")
fs = 30720000
sdr.sample_rate = fs
sdr.rx_lo = 2000000000
sdr.tx_lo = 2000000000
sdr.rx_hardwaregain_chan0 = 60
sdr.rx_enabled_channels = [0]
sdr.rx_buffer_size = 32768

# 2. 发射 5MHz 单音
sdr.tx_enabled_channels = [0]
fc = 5000000
N = 1024
t = np.arange(N)/fs
i = np.cos(2*np.pi*fc*t) * 16384
q = np.sin(2*np.pi*fc*t) * 16384
iq = i + 1j*q

sdr.tx_destroy_buffer()
sdr.tx_cyclic_buffer = True
sdr.tx(iq)

# 3. 画图修正版
plt.ion()
fig, ax = plt.subplots(figsize=(12,6))
time_offset = 0.0  # 累积时间戳

while True:
    x = sdr.rx()
    sig = x  # 注意：FMComms5返回是列表，取通道0
    
    f, t_spec, Sxx = signal.spectrogram(
        sig,
        fs=fs,
        nperseg=512,
        noverlap=256,
        window='hann',
        scaling='density',
        return_onesided=False  # ✅ 强制输出双边谱 [-fs/2, fs/2]
    )
    
    # 频率轴平移到 [-fs/2, fs/2]
    f_shifted = np.fft.fftshift(f)
    Sxx_shifted = np.fft.fftshift(Sxx, axes=0)
    
    ax.cla()
    # 时间轴累积，避免每次重置
    t_spec_shifted = t_spec + time_offset
    
    ax.pcolormesh(f_shifted, t_spec_shifted, 10*np.log10(Sxx_shifted.T), 
                  cmap='jet', shading='gouraud')
    ax.set_xlim(-fs//2, fs//2)  # 现在和数据匹配
    ax.set_ylim(time_offset, time_offset + t_spec[-1])
    ax.set_title("Complex IQ Spectrogram (5MHz Signal)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (s)")
    plt.draw()
    plt.pause(0.05)
    time_offset += t_spec[-1]