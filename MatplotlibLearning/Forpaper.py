import matplotlib.pyplot as plt
import numpy as np
import os
import torch



def getdata(dir_path):
    file_name_list = os.listdir(dir_path)
    file_path_list =[]

    for file_name in file_name_list :
        file_path_list.append(os.path.join(dir_path,file_name))

    axis_list = []

    for  file_path in file_path_list:
        axis = []
        with open(file_path) as file_object:
            for line in file_object:
                x , y = line.split(',')
                y = y.replace("\n", "")
                axis.append((x,y))
        axis_list.append(axis)

        #print(len(axis))



    return axis_list

def  drawAxis(dirpath):


    axis_file_list = getdata(dirpath)
    a = 0
    for axis_list in axis_file_list:
        x_list = []
        y_list = []
        for i in range(800):
            if i <len(axis_list):
                x, y = axis_list[i]
            else:
                x = 0
                y = 0
            x_list.append(float(x))
            y_list.append(float(y))
        plt.xlabel('time')
        plt.ylabel('axis')

        yLine = np.linspace(0,800,800)

        plt.plot(yLine,x_list,color='red',label = 'Xaxis')
        plt.plot(yLine,y_list,color='blue',label = 'Yaxis')
        plt.legend(loc=4)
        print(a)
        plt.show()
        a = a+1
        #plt.savefig('squares_plot.png')

def creatDateSet(dirpath):
    axis_file_list = getdata(dirpath)
    dataset =[]
    for axis_list in axis_file_list:
        axis=[]
        for i in range(800):
            if i <len(axis_list):
                x, y = axis_list[i]
            else:
                x = 0
                y = 0
            axis.append((float(x),float(y)))

        dataset.append(axis)
    tensor_dataset = torch.tensor(dataset)

    return tensor_dataset

def getFileNum(dirpath):
    file_name_list = os.listdir(dirpath)
    return len(file_name_list)


cheat_dir_path = 'E:/unityWorkSpace/FPSTest/Assets/dataset/axis_cheat'
dir_path = 'E:/unityWorkSpace/FPSTest/Assets/dataset/axis'
#print(getFileNum(cheat_dir_path))
#print(getFileNum(cheat_dir_path)/11)

print(creatDateSet(cheat_dir_path)[0].shape)