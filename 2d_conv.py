import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from my_conv import my_2d_jit
# 读取灰度图像
img = Image.open('input/metro.jpg').convert('L')  # 使用PIL打开并转为灰度图
img = np.array(img)

# 转换为Tensor
img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度

# 添加高斯噪声（比椒盐噪声更弱）
def add_gaussian_noise(image, mean=0, std=30):
    noisy_img = image.clone()
    noise = torch.normal(mean, std, size=image.size())  # 生成高斯噪声
    noisy_img += noise
    noisy_img = torch.clamp(noisy_img, 0, 255)  # 保证像素值在有效范围内
    return noisy_img

# 卷积操作
def apply_convolution(image, kernel):
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
    return F.conv2d(image, kernel_tensor, padding=kernel.shape[0] // 2)
def get_gaussian_kernel_2d(ksize, sigma):
    k_h, k_w = ksize
    # 确保核大小是奇数，以便有明确的中心
    if k_h % 2 == 0 or k_w % 2 == 0:
        raise ValueError("Kernel dimensions must be odd numbers.")
    x_ax = np.linspace(-(k_w - 1) / 2., (k_w - 1) / 2., k_w)
    y_ax = np.linspace(-(k_h - 1) / 2., (k_h - 1) / 2., k_h)
    xv, yv = np.meshgrid(x_ax, y_ax)
    kernel = np.exp(-0.5 * (np.square(xv) + np.square(yv)) / np.square(sigma))
    normalized_kernel = kernel / np.sum(kernel)
    return normalized_kernel

# 去噪实验
def denoising_experiment():
    noisy_img_tensor = add_gaussian_noise(img_tensor).squeeze().detach().numpy()  # 使用高斯噪声
    kernels=[]
    names=["avg_len5","gaussian_5","gaussian_10"]
    kernels.append(np.ones((5, 5), np.float32) / 25)
    kernels.append(get_gaussian_kernel_2d((5,5),5))
    kernels.append(get_gaussian_kernel_2d((5,5),10))
    for i in range(len(kernels)):
        denoised_img_tensor = my_2d_jit(noisy_img_tensor, kernels[i])
    
        # 转换回numpy数组进行显示
        original_img = img  # 原图
        noisy_img = noisy_img_tensor#.squeeze().detach().numpy()
        #denoised_img = denoised_img_tensor.squeeze().detach().numpy()
        denoised_img=denoised_img_tensor
        # 创建带有高分辨率的子图，设置dpi，确保显示清晰
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=100)  # 设置高分辨率和合适的尺寸
        
        axs[0].imshow(original_img, cmap='gray')
        axs[0].set_title('Original Image', fontsize=12)
        axs[0].axis('off')
        
        axs[1].imshow(noisy_img, cmap='gray')
        axs[1].set_title('Noisy Image', fontsize=12)
        axs[1].axis('off')
        
        axs[2].imshow(denoised_img, cmap='gray')
        axs[2].set_title('Denoised Image', fontsize=12)
        axs[2].axis('off')
        
        plt.tight_layout(pad=3.0)  # 增加子图之间的空隙
        plt.savefig(f'output/image/denoising_{names[i]}.png', dpi=300)  # 保存为300dpi高分辨率图像
        plt.close()

# 边缘提取实验
def edge_detection_experiment():
    names=["Vertival","Horizontal","Laplace"]
    kernels=[]
    kernels.append(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) )
    kernels.append(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32).transpose())
    kernels.append(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32))  # 边缘检测核
    fig, axs = plt.subplots(1,len(kernels)+1 , figsize=(15, 5), dpi=100)  # 设置高分辨率和合适的尺寸
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image', fontsize=12)  # 调整字体大小
    axs[0].axis('off')
        
    for i in range(len(kernels)):
        edge_img_tensor = my_2d_jit(img_tensor.squeeze().detach().numpy() , kernels[i],keep=True)
        # 转换回numpy数组进行显示
        edge_img=edge_img_tensor+img_tensor.squeeze().detach().numpy() 
        # 创建带有高分辨率的子图，设置dpi，确保显示清晰
        axs[i+1].imshow(edge_img, cmap='gray')
        axs[i+1].set_title(f'Edge Detection: {names[i]}', fontsize=12)  # 调整字体大小
        axs[i+1].axis('off')
        
    plt.tight_layout(pad=3.0)  # 增加子图之间的空隙
    plt.savefig('output/image/edge_detection.png', dpi=300)
    plt.close()

# 执行实验
denoising_experiment()
edge_detection_experiment()
