import adi
import numpy as np
import matplotlib.pyplot as plt
# 1. 创建设备
sdr = adi.FMComms5(uri="ip:192.168.1.10")

# 2. 基础配置
sdr.sample_rate = 30720000  # 采样率 30.72M
sdr.tx_lo = 2000000000      # 发射本振 2G
sdr.rx_lo = 2000000000      # 接收本振 2G
sdr.tx_cyclic_buffer = True # 循环发射

# 3. 只打开 1 个 TX 通道：TX0
sdr.tx_enabled_channels = [0]
sdr.rx_enabled_channels = [0]
# ==============================
# 4. 生成 10MHz 复数信号（I=10M，Q=10M）
# ==============================
fc = 10000000  # 10MHz！！！
fc2=5000000
N = 1024
ts = 1 / sdr.sample_rate
t = np.arange(0, N * ts, ts)

# I路 = 10MHz 余弦
i = np.cos(2 * np.pi * t * fc) * 16384
# Q路 = 10MHz 正弦
q = np.sin(2 * np.pi * t * fc2) * 16384

# 组合成复数
iq = i + 1j * q

# 5. 发射（单通道直接传复数即可）
sdr.tx(iq)


data=sdr.rx()

sig = data



# sig=np.imag(data)

print(sig)

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