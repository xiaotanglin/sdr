#这是最早采集，只是采集IQ数据


import adi
import numpy as np
import matplotlib.pyplot as plt

# 连接
sdr = adi.FMComms5(uri="ip:192.168.1.30")

# 配置
sdr.sample_rate = 30720000
sdr.rx_lo = 2450000000
sdr.rx_rf_bandwidth = 18000000
sdr.rx_enabled_channels = [0]
sdr.rx_hardwaregain_chan0 = 70
sdr.rx_buffer_size = 1024

# 采集
print("采集数据...")
data = sdr.rx()

# 强制转数组（终极修复）
iq_array = np.array([data]).flatten()

print("数据长度:", len(iq_array))
print("前10个原始IQ:\n", iq_array[:10])

# 分离 I/Q
I = np.real(iq_array)
Q = np.imag(iq_array)

# 画图
plt.figure(figsize=(12,6))

plt.subplot(211)
plt.plot(I, linewidth=0.5)
plt.title("I Channel")
plt.grid(True)

plt.subplot(212)
plt.plot(Q, linewidth=0.5)
plt.title("Q Channel")
plt.grid(True)

plt.tight_layout()
plt.show()