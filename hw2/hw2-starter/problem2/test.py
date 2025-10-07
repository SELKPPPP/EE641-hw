import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import matplotlib.pyplot as plt  

if __name__ == '__main__':
    data_path = os.path.join('data/drums', 'patterns.npz')
    data = np.load(data_path)
    print(data.files)  # 查看包含的数组名称
    
