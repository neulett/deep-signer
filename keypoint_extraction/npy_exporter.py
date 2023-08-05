import numpy as np
import pandas as pd
import os

path = 'C:/Users/ILIXYENILXY/Desktop/deep_signer/local_dir/keypoints_data'
df = pd.read_excel('C:/Users/ILIXYENILXY/Desktop/deep_signer/local_dir/KETI-2017-SL-Annotation-v2_1.xlsx')
df = df.sort_values('번호')

def create_array(size, value):
    arr = np.array([value] * size)
    return arr

label_path = 'C:/Users/ILIXYENILXY/Desktop/deep_signer/local_dir/label_path'
emp_lst = []
label = np.array(emp_lst)

for filename in os.listdir(path):
    file_name = filename[0:18]
    file_path = os.path.join(path, filename)
    array = np.load(file_path)

    get_index = int(filename[13:18])   # csv열에 맞게끔 인덱싱번호 생성

    result = df.iloc[get_index]     # get_index를 통한 열 인덱싱
    fill_val = result['한국어']     # 같은 열의 라벨 인덱싱

    length_array = array.shape[0]   # 
    if length_array > 0:
        arr = create_array(length_array, fill_val)
        np.save(f'{label_path}/{file_name}_label', arr)
    else:
        break

file_list = os.listdir(label_path)
npy_files = [f for f in file_list if f.endswith('.npy')]

concatenated_array = np.concatenate([np.load(os.path.join(label_path, file)) for file in npy_files])
np.save('concatnated_label', concatenated_array)