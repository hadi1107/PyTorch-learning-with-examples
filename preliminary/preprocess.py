import os
import pandas as pd
import torch

def create_data():
    # 当exist_ok=True时，如果目录已经存在，os.makedirs()函数不会引发异常，而是静默地继续执行。
    # 当exist_ok=False时，如果目录已经存在，os.makedirs()函数将引发FileExistsError异常。
    os.makedirs(os.path.join('..','data'),exist_ok = True)
    data_file = os.path.join('..','data','house_tiny.csv')
    with open(data_file,'w') as f:
        # 房间数量、街道小巷类型和房价
        f.write('NumRooms,Alley,Price\n')
        f.write('NA,Pave,127500\n')
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')

def read_data():
    data = pd.read_csv('D:\\torch_study\\data\\house_tiny.csv')
    print(data)
    print()
    return data

def interpolation(data):
    # 利用插值法补齐num_rooms的NAN，并将alley转换为数值类型
    inputs,outputs = data.iloc[:,0:2], data.iloc[:,2]
    inputs = inputs.fillna(inputs.mean(numeric_only=True))
    inputs = pd.get_dummies(inputs, dummy_na =True)
    X = torch.tensor(inputs.values)
    y = torch.tensor(outputs.values)
    print(X)
    print(y)
    return X,y

if __name__ == "__main__":
    create_data()
    data = read_data()
    X,y = interpolation(data)
