import numpy as np

# 加载文件
data = np.load('results/value_function.npz')

# 查看里面有哪些数组（键名）
print("Keys:", data.files)

# 查看具体数据
for key in data.files:
    print(f"\n--- {key} ---")
    print(data[key])