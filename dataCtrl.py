
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class MyData(Dataset):

    train_cheat_dir_path = 'dataset/train/cheat'
    train_dir_path = 'dataset/train/normal'
    test_cheat_dir_path = 'dataset/test/cheat'
    test_dir_path = 'dataset/test/normal'

    def __init__(self,nntype,datasetType):
        if (datasetType =='train' ):
            #train set
            train_axis_cheat = creatDateSet(train_cheat_dir_path, nntype)
            cheat_lable =torch.ones(train_axis_cheat.shape[0])

            train_axis = creatDateSet(train_dir_path, nntype)
            normal_label =torch.zeros(train_axis.shape[0])

            self.axis_set = torch.cat((train_axis_cheat, train_axis), dim=0)
            self.label_set = torch.cat((cheat_lable, normal_label), dim=0)
            self.label_set = torch.Tensor(self.label_set).long()

        elif(datasetType =='test' ):
            # test set
            test_axis_cheat = creatDateSet(test_cheat_dir_path, nntype)
            test_cheat_lable = torch.ones(test_axis_cheat.shape[0])

            test_axis = creatDateSet(train_dir_path, nntype)
            test_normal_label = torch.zeros(test_axis.shape[0])

            self.axis_set = torch.cat((test_axis_cheat, test_axis), dim=0)
            self.label_set = torch.cat((test_cheat_lable, test_normal_label), dim=0)
            self.label_set = torch.Tensor(self.label_set).long()


    def __getitem__(self, index):

        return self.axis_set[index],self.label_set[index]

    def __len__(self):
        return len(self.axis_set)



def getdata(dir_path,draw =False):
    file_name_list = os.listdir(dir_path)
    file_path_list =[]

    for file_name in file_name_list :
        file_path_list.append(os.path.join(dir_path,file_name))

    axis_list = []
    axis_len_list = []
    for  file_path in file_path_list:
        axis = []
        with open(file_path) as file_object:
            for line in file_object:
                x , y = line.split(',')
                y = y.replace("\n", "")
                axis.append((x,y))
        axis_list.append(axis)
        axis_len_list.append(len(axis))
    if draw:
        return axis_len_list



    return axis_list ,file_path_list

def  drawAxis(dirpath):


    axis_file_list, file_path_list = getdata(dirpath)
    a = 0
    for num in range(len(axis_file_list)):
        x_list = []
        y_list = []
        for i in range(800):
            if i <len(axis_file_list[num]):
                x, y = axis_file_list[num][i]
            else:
                x = 0
                y = 0
            x_list.append(float(x))
            y_list.append(float(y))
        plt.xlabel(file_path_list[num])
        plt.ylabel('axis')

        yLine = np.linspace(0,800,800)

        plt.plot(yLine,x_list,color='red',label = 'Xaxis')
        plt.plot(yLine,y_list,color='blue',label = 'Yaxis')
        plt.legend(loc=4)
        print(a)
        plt.show()
        a = a+1
        #plt.savefig('squares_plot.png')

def creatDateSet(dirpath,nntype):
    axis_file_list, file_path_list = getdata(dirpath)

    if(nntype == 'RNN'):
        pass

    elif(nntype == 'single'):
        dataset = []
        for axis_list in axis_file_list:
            axis = []
            for i in range(800):
                if i < len(axis_list):
                    x, y = axis_list[i]
                else:
                    x = 0
                    y = 0
                axis.append((float(x), float(y)))
            axis = torch.tensor(axis)
            axis = torch.reshape(axis,[1,1,40,40])
            dataset.append(axis)

        for i in range(len(dataset)):
            if i == 0:
                continue
            if i == 1:
                tensor_dataset =torch.cat((dataset[i-1],dataset[i]),dim=0)
            else:
                tensor_dataset = torch.cat((tensor_dataset, dataset[i]), dim=0)

        return tensor_dataset,file_path_list

    elif(nntype == 'two'):
        dataset = []
        for axis_list in axis_file_list:
            Xaxis = []
            Yaxis = []

            for i in range(900):
                if i < len(axis_list):
                    x, y = axis_list[i]
                else:
                    x = 0
                    y = 0
                Xaxis.append(float(x))
                Yaxis.append(float(y))
            Xaxis = torch.tensor(Xaxis)
            Xaxis = torch.reshape(Xaxis, [1, 30, 30])
            Yaxis = torch.tensor(Yaxis)
            Yaxis = torch.reshape(Yaxis, [1, 30, 30])
            axis = torch.cat((Xaxis,Yaxis),dim= 0)
            axis = torch.unsqueeze(axis,0)
            dataset.append(axis)
        for i in range(len(dataset)):
            if i == 0:
                continue
            if i == 1:
                tensor_dataset =torch.cat((dataset[i-1],dataset[i]),dim=0)
            else:
                tensor_dataset = torch.cat((tensor_dataset, dataset[i]), dim=0)


        return tensor_dataset, file_path_list #[1,2,30,30]

def getFileNum(dirpath):
    file_name_list = os.listdir(dirpath)
    return len(file_name_list)

def drawAxisNum():
    AxisNum = []
    for num in getdata(train_cheat_dir_path,True):
        AxisNum.append(num)
    for num in getdata(train_dir_path,True):
        AxisNum.append(num)
    plt.bar(range(len(AxisNum)), AxisNum)
    plt.show()
    pass


train_cheat_dir_path = 'dataset/train/cheat'
train_dir_path = 'dataset/train/normal'
test_cheat_dir_path = 'dataset/test/cheat'
test_dir_path = 'dataset/test/normal'
verify_cheat_dir_path  ='dataset/ verify/cheat'
verify_dir_path  ='dataset/ verify/normal'
#print(getFileNum(verify_cheat_dir_path))
#drawAxis(verify_cheat_dir_path)
#drawAxis(test_cheat_dir_path)
#print(getFileNum(cheat_dir_path)/11)

# testdata1 = creatDateSet(train_cheat_dir_path,nntype = 'single')
# testdata2 = creatDateSet(train_dir_path,nntype = 'single')
# testdata = torch.cat((testdata1,testdata2),dim=0)


# train_axis_cheat = creatDateSet(train_cheat_dir_path, nntype='two')
# print(train_axis_cheat.shape[0])
# cheat_lable =torch.ones(int(train_axis_cheat.shape[0]))
#
# train_axis = creatDateSet(train_dir_path, nntype='two')
# print(int(train_axis.shape[0]))
# normal_label =torch.zeros(int(train_axis.shape[0]))
#
# train_axis_set = torch.cat((train_axis_cheat, train_axis), dim=0)
# train_label_set = torch.cat((cheat_lable, normal_label), dim=0)
#
# print(train_label_set)
# print(train_label_set.shape)

# train_axis_cheat = creatDateSet(train_cheat_dir_path, nntype='two')
# print(train_axis_cheat[0].shape)

# drawAxisNum()

