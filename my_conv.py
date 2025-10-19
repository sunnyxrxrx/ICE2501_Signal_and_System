import numpy as np
import torch
from numba import jit
import torch.nn.functional as F

def my_1d(signal,kernel):
    len1=len(signal)
    len2=len(kernel)
    len3=len1+len2-1
    result=np.zeros(len3)
    signal=np.pad(signal,(len2-1,len2-1),"constant",constant_values=0)
    print(f"输出长度{len(result)}")
    for i in range(0,len3):
        for j in range (0,len2):
            result[i]+=kernel[len2-j-1]*signal[i+j]
    return result   
def my_1d_fast(signal,kernel):
    len1=len(signal)
    len2=len(kernel)
    len3=len1+len2-1
    kernel_flipped = kernel[::-1] 
    result=np.zeros(len3)
    signal_padded=np.pad(signal,(len2-1,len2-1),"constant",constant_values=0)
    print(f"输出长度{len(result)}")
    for i in range (0,len3):
        signal_window = signal_padded[i : i + len2]
        result[i]=np.dot(signal_window,kernel_flipped)
    return result   


@jit(nopython=True) # 加上这行魔法装饰器
def my_1d_jit(signal, kernel):
    len1 = len(signal)
    len2 = len(kernel)
    len3 = len1 + len2 - 1
    kernel_flipped = kernel[::-1] 
    result = np.zeros(len3)
    pad_width = len2-1
    signal_padded = np.zeros(len1 + 2 * pad_width)
    signal_padded[pad_width : pad_width + len1] = signal
    for i in range(len3):
        signal_window = signal_padded[i : i + len2]
        result[i] = np.dot(signal_window, kernel_flipped)
    return result

def my_1d_torch(signal, kernel):
    kernel_flip = kernel[::-1].copy()
    kernel_tensor = torch.tensor(kernel_flip, dtype=torch.float64).unsqueeze(0).unsqueeze(0)  # 转换为tensor
    signal_tensor = torch.tensor(signal, dtype=torch.float64).unsqueeze(0).unsqueeze(0)  # 转换为tensor
    result = F.conv1d(signal_tensor, kernel_tensor, padding=kernel.shape[0]-1)
    return result.squeeze().detach().numpy()

def my_2d(signal,kernel):
    lenm=np.size(signal,axis=0)# m是行数，n是列数
    lenn=np.size(signal,axis=1)
    m=np.size(kernel,axis=0)
    n=np.size(kernel,axis=1)
    signal=np.pad(signal,((m-1,m-1),(n-1,n-1)),'constant',constant_values=0)
    result=np.zeros((lenm+m-1,lenn+n-1))
    for i in range(0,lenn+n-1):
        for j in range(0,lenm+m-1):
            for x in range(0,m):
                for y in range(0,n):
                    result[i][j]+=signal[i+x][j+y]*kernel[m-x-1][n-y-1]
    return result

@jit(nopython=True)
def my_2d_jit(signal,kernel,keep=False):
    lenm,lenn=signal.shape # 信号的大小
    m,n=kernel.shape # 卷积核的大小
    assert (m%2==1 and n%2==1)
    result=np.zeros((lenm+m-1,lenn+n-1))
    signal_padded=np.zeros((lenm+2*m-2,lenn+2*n-2))
    signal_padded[m-1:m+lenm-1,n-1:n+lenn-1]=signal
    kernel_flipped = np.flip(kernel)
    for i in range(lenm+m-1):
        for j in range(lenn+n-1):
            signal_window = signal_padded[i : i + m, j : j + n]
            result[i, j] = np.sum(signal_window * kernel_flipped)
    if keep:
        return result[(m-1)//2:(m-1)//2+lenm,(n-1)//2:(n-1)//2+lenn]
    return result

def my_2d_torch(signal, kernel,keep=False):
    m,n=kernel.shape
    assert (m%2==1 and n%2==1)
    kernel_flip = np.flip(kernel).copy()
    kernel_tensor = torch.tensor(kernel_flip, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    signal_tensor = torch.tensor(signal, dtype=torch.float64).unsqueeze(0).unsqueeze(0)  # 转换为tensor
    if keep:
        return F.conv2d(signal_tensor, kernel_tensor, padding=((m-1)//2,(n-1)//2)).squeeze().detach().numpy()

    return F.conv2d(signal_tensor, kernel_tensor, padding=(m-1,n-1)).squeeze().detach().numpy()
    
