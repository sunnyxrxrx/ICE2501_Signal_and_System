import numpy as np
import torch
import torch.nn.functional as F
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from my_conv import my_1d_fast,my_1d,my_1d_jit,my_1d_torch
import time 
# 加载语音文件
def load_audio(file_path):
    signal, sr = librosa.load(file_path, sr=None)  # 加载音频信号，并保持原始采样率
    return signal, sr

# 保存音频文件
def save_audio(file_path, signal, sr):
    sf.write(file_path, signal, sr)  # 使用soundfile保存音频

# 卷积操作
def apply_convolution(signal, kernel):
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 转换为tensor
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 转换为tensor
    # 填充使卷积输出的长度和输入一样
    conv_signal = F.conv1d(signal_tensor, kernel_tensor, padding=kernel.shape[0] // 2)
    return conv_signal.squeeze().detach().numpy()

 

def sharpening_experiment():
    signal, sr = load_audio('input/SJTU_Ring.mp3')
    # 锐化卷积核
    kernel1 = np.array([-1,2,-20],dtype=np.float64)  # 锐化卷积核
    kernel2=np.array([1/3, 1/3, 1/3],dtype=np.float64)
    times={}
    names=["my_1d","my_1d_fast","my_1d_jit"]
    for j,method in enumerate([my_1d,my_1d_fast,my_1d_jit]):
        start_time=time.perf_counter()
        for i in range(10):
            sharpened_signal = method(signal, kernel1) 
            down_signal=method(signal,kernel2)
        end_time=time.perf_counter()
        times[names[j]]=end_time-start_time
        print(f"{names[j]}完成")
    return times

# 执行实验
print(sharpening_experiment())