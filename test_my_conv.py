import my_conv
import numpy as np
from my_conv import my_1d_jit,my_1d_torch,my_2d_jit,my_2d_torch



signal1 = np.random.rand(10)
kernel1 = np.random.rand(3) 
result_jit = my_1d_jit(signal1, kernel1)
result_torch = my_1d_torch(signal1, kernel1)

print("JIT结果: ", result_jit)
print("PyTorch结果:", result_torch)
print("结果是否一致:", np.allclose(result_jit, result_torch))

for i in [True,False]:
    signal2=np.random.rand(10,10)
    kernel2=np.random.rand(3,3)
    result_jit1 = my_2d_jit(signal2, kernel2,keep=i)
    result_torch1 = my_2d_torch(signal2, kernel2,keep=i)
    print("JIT结果: ",result_jit1)
    print("PyTorch结果:",result_torch1)
    print("结果是否一致:", np.allclose(result_jit1,result_torch1))