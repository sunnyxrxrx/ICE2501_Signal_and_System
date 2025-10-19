import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from my_conv import my_2d_jit

def canny_edge_detection(image, low_threshold=50, high_threshold=150, sigma=1.4):
    """
    完整的Canny边缘检测实现
    
    参数:
        image: 输入灰度图像 (numpy数组)
        low_threshold: 低阈值
        high_threshold: 高阈值
        sigma: 高斯滤波标准差
    
    返回:
        edges: 检测到的边缘图像
    """
    
    # 步骤1: 高斯滤波降噪
    gaussian_kernel = create_gaussian_kernel(size=5, sigma=sigma)
    smoothed = my_2d_jit(image, gaussian_kernel, keep=True)
    
    # 步骤2: 计算梯度幅值和方向
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)
    
    gradient_x = my_2d_jit(smoothed, sobel_x, keep=True)
    gradient_y = my_2d_jit(smoothed, sobel_y, keep=True)
    
    # 计算梯度幅值和方向
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    
    # 步骤3: 非极大值抑制
    nms_result = non_maximum_suppression(gradient_magnitude, gradient_direction)
    
    # 步骤4和5: 双阈值检测和边缘连接
    edges = double_threshold_and_hysteresis(nms_result, low_threshold, high_threshold)
    
    return edges, gradient_magnitude, smoothed


def create_gaussian_kernel(size=5, sigma=1.4):
    """
    创建高斯卷积核
    """
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # 归一化
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def non_maximum_suppression(gradient_magnitude, gradient_direction):
    """
    非极大值抑制 - 使边缘变细
    """
    h, w = gradient_magnitude.shape
    suppressed = np.zeros_like(gradient_magnitude)
    
    # 将角度转换到0-180度
    angle = gradient_direction * 180.0 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q = 255
            r = 255
            
            # 根据梯度方向确定要比较的邻居像素
            # 0度: 水平方向
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            # 45度: 对角线方向
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            # 90度: 垂直方向
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            # 135度: 对角线方向
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]
            
            # 如果当前像素是局部最大值，则保留
            if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                suppressed[i, j] = gradient_magnitude[i, j]
            else:
                suppressed[i, j] = 0
    
    return suppressed


def double_threshold_and_hysteresis(image, low_threshold, high_threshold):
    """
    双阈值检测和边缘连接（滞后阈值）
    """
    h, w = image.shape
    
    # 初始化结果图像
    result = np.zeros_like(image, dtype=np.uint8)
    
    # 定义强边缘、弱边缘和非边缘
    weak = 75
    strong = 255
    
    # 根据阈值分类
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))
    
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    
    # 边缘连接（滞后）
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if result[i, j] == weak:
                # 检查8邻域是否有强边缘
                if ((result[i + 1, j - 1] == strong) or 
                    (result[i + 1, j] == strong) or 
                    (result[i + 1, j + 1] == strong) or
                    (result[i, j - 1] == strong) or 
                    (result[i, j + 1] == strong) or
                    (result[i - 1, j - 1] == strong) or 
                    (result[i - 1, j] == strong) or 
                    (result[i - 1, j + 1] == strong)):
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    
    return result


# 使用示例
def demo_canny(image):
    """
    演示Canny边缘检测的各个步骤
    """
    # 执行Canny边缘检测
    edges, gradient_mag, smoothed = canny_edge_detection(
        image, 
        low_threshold=50, 
        high_threshold=150, 
        sigma=1.4
    )
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('original')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(smoothed, cmap='gray')
    plt.title('Gaussian Denoise')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(gradient_mag, cmap='gray')
    plt.title('gradient')
    plt.axis('off')
    
    
    # 不同参数对比
    edges_low = canny_edge_detection(image, 30, 90, 1.4)[0]
    edges_high = canny_edge_detection(image, 100, 200, 1.4)[0]
    
    plt.subplot(2, 2, 4)
    plt.imshow(edges_low, cmap='gray')
    plt.title('final result')
    plt.axis('off')
    
    # plt.subplot(2, 3, 6)
    # plt.imshow(edges_high, cmap='gray')
    # plt.title('(100, 200)')
    # plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return edges


# 调用方式：
# edges = canny_edge_detection(your_image, low_threshold=50, high_threshold=150)
# 或者查看完整过程：
img = Image.open('input/123.jpg').convert('L')  # 使用PIL打开并转为灰度图
img = np.array(img)
demo_canny(img)