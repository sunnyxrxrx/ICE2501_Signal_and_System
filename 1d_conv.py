import numpy as np
import torch
import torch.nn.functional as F
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from my_conv import my_1d_fast,my_1d
# 加载语音文件
def load_audio(file_path):
    signal, sr = librosa.load(file_path, sr=None)  # 加载音频信号，并保持原始采样率
    return signal, sr

# 保存音频文件
def save_audio(file_path, signal, sr):
    sf.write(file_path, signal, sr)  # 使用soundfile保存音频

# 添加噪声
def add_noise(signal, noise_factor=0.01):
    noise = np.random.normal(0, noise_factor, signal.shape)
    noisy_signal = signal + noise
    return noisy_signal

# 卷积操作
def apply_convolution(signal, kernel):
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 转换为tensor
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 转换为tensor
    # 填充使卷积输出的长度和输入一样
    conv_signal = F.conv1d(signal_tensor, kernel_tensor, padding=kernel.shape[0] // 2)
    return conv_signal.squeeze().detach().numpy()

def gaussian_kernel(size, sigma):
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / np.sum(kernel)


# 去噪实验
def denoising_experiment():
    signal, sr = load_audio('input/SJTU_Ring.mp3')
    noisy_signal = add_noise(signal, noise_factor=0.1)  # 添加噪声
    save_audio('output/noisy_signal.wav', noisy_signal, sr)  # 保存加噪后的信号
    kernels=[]
    # 平均模糊核
    kernels.append(np.ones(1000) / 1000) 
    kernels.append(np.ones(10) / 10)
    # 高斯核
    kernels.append(gaussian_kernel(1000, 5))  # 可根据需要调整 sigma 值
    kernels.append(gaussian_kernel(1000, 10))
    kernels.append(gaussian_kernel(1000, 20))  
    names=["avg_len1000","avg_len10","gaussian_5","gaussian_10","gaussian_20"]
    for i in range(len(kernels)):
        denoised_signal = my_1d_fast(noisy_signal, kernels[i])
        
        # 根据原始信号的最大幅值对去噪信号进行缩放
        max_original = np.max(np.abs(signal))  # 原始信号的最大幅值
        max_denoised = np.max(np.abs(denoised_signal))  # 去噪后信号的最大幅值
        scaled_denoised_signal = denoised_signal * (max_original / max_denoised)  # 缩放去噪信号
        
        save_audio(f'output/denoised_{names[i]}.wav', scaled_denoised_signal, sr)  # 保存去噪后的信号

# 锐化实验
def sharpening_experiment():
    signal, sr = load_audio('input/SJTU_Ring.mp3')
    
    # 锐化卷积核
    kernel = np.array([0, -1, -1,  5, -1, -1, 0])  # 锐化卷积核
    sharpened_signal = my_1d_fast(signal, kernel)  
    save_audio('output/sharpened.wav', sharpened_signal, sr)  # 保存锐化后的信号

# 执行实验
denoising_experiment()
sharpening_experiment()
