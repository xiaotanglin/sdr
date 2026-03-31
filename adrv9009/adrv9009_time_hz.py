import adi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# Create radio
sdr = adi.adrv9009(uri="ip:192.168.1.10")
sdr.rx_enabled_channels=[0]
sdr.tx_enabled_channels = [0]
sdr.trx_lo=2_000_000_000
sdr.gain_control_mode_chan0="slow_attack"
sdr.rx_buffer_size=1024
print(sdr.rx_sample_rate)
fs=sdr.rx_sample_rate

#tx
fs = 122_880_000
fc = 5000000
N = 1024
t = np.arange(N)/fs
i = np.cos(2*np.pi*fc*t) * 16384
q = np.sin(2*np.pi*fc*t) * 16384
iq = i + 1j*q

sdr.tx_destroy_buffer()
sdr.tx_cyclic_buffer = True
sdr.tx(iq)



# 3. 画图：横轴=时间，纵轴=频率（瀑布图）
plt.rcParams['figure.figsize'] = (12, 6)
plt.ion()
fig, ax = plt.subplots()

# 存储历史频谱，形成滚动瀑布图
history = []
max_frames = 80  # 控制瀑布图长度

while True:
    x = sdr.rx()
    sig = x # 通道0的IQ复数信号

    # 计算时频图（双边谱）
    f, t_spec, Sxx = signal.spectrogram(
        sig,
        fs=fs,
        nperseg=1024,
        noverlap=512,
        window='hann',
        return_onesided=False  # 关键：复数IQ必须用双边谱
    )

    # 频率轴移到 [-fs/2, fs/2]
    f = np.fft.fftshift(f) / 1e6  # 转 MHz
    Sxx = np.fft.fftshift(Sxx, axes=0)

    # 取当前最新一帧频谱
    current_spec = 10 * np.log10(Sxx.mean(axis=1))  # 转dB

    # 加入历史
    history.append(current_spec)
    if len(history) > max_frames:
        history.pop(0)

    # 绘制：横轴=时间，纵轴=频率
    ax.cla()
    ax.imshow(
        np.array(history).T,       # 转置 → 时间横轴，频率纵轴
        aspect='auto',
        cmap='jet',
        origin='lower',           # 低频在下，高频在上
        extent=[0, max_frames, f.min(), f.max()]
    )

    ax.set_title("SDR time-hz (x=time,y=freq)", fontsize=14)
    ax.set_xlabel("time", fontsize=12)
    ax.set_ylabel("freq (MHz)", fontsize=12)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)
