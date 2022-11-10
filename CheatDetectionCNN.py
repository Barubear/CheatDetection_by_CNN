import random
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sigmoid, ReLU, Softmax
from torch.utils.data import DataLoader
from dataCtrl import MyData,creatDateSet,getFileNum,drawAxis



class SingleChannelCNN(nn.Module):
    nnName = 'SingleChannelCNN'

    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(

            Conv2d(1, 40, 3, padding=1, stride=1),
            ReLU(),
            MaxPool2d(2),

            Conv2d(40, 80, 3, padding=1, stride=1),
            ReLU(),
            MaxPool2d(2),

            Conv2d(80, 80, 3, padding=1, stride=1),
            ReLU(),
            MaxPool2d(2),

            Flatten(),

            Linear(2000,400),
            Linear(400,2)


        )

    def forward(self, x):
       return self.model1(x)

class TwoChannelCNN(nn.Module):
    nnName = 'TwoChannelCNN'
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            Conv2d(2, 30, 3, padding=1, stride=1),#30,30,30
            MaxPool2d(2),#30, 15, 15
            Conv2d(30, 60, 3, padding=1, stride=1),#60, 15, 15
            MaxPool2d(2),# 60, 7, 7
            Conv2d(60, 60, 3, padding=1, stride=1),#60, 7, 7
            MaxPool2d(2),#60, 3, 3
            Flatten(),
            Linear(540,60),
            Linear(60,2)
        )

    def forward(self,x):

       return self.model1(x)

#writepoch: To decide The lenth of loss char’s Y axis , == epoch/writepoch
def train(usedNN, train_dataloder, test_dataloder,epoch, loss_fn, optimizer,Issave = False,saveEpoch = 10,writepoch = 1):
#训练次数
    total_train_step = 0
#测试次数


    loss_list = []

    yLine = np.linspace(0,int(epoch/writepoch),int(epoch/writepoch))


    for i in range(epoch):

        #train
        for data in train_dataloder:
            axis ,targets = data
            axis =axis.cuda()
            targets =targets.cuda()
            output = usedNN(axis)
            loss = loss_fn(output,targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step +1

        #test

        with torch.no_grad():

            total_test_loss = 0
            total_test_accuracy = 0
            test_step = 0

            for data in test_dataloder:
                axis, targets = data
                axis = axis.cuda()
                targets = targets.cuda()
                output = usedNN(axis)
                loss = loss_fn(output, targets)




                total_test_loss = total_test_loss + loss.cpu()



        loss_list.append(total_test_loss)


        if ((i+1) % saveEpoch == 0):
            #loss_list.append(total_test_loss)
            print("epoch:{},testloss = {}".format(i + 1, total_test_loss))
            if Issave:
                torch.save(usedNN, '{}_{}.pth'.format(usedNN.nnName, i+1))
                print('model is saved ')







    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.plot(yLine, loss_list, color='red', label='loss')

    plt.show()

def SingleChannelCNNtain(epoch,Issave = False,saveEpoch = 10,writepoch = 1):

    singlechannelCNN = SingleChannelCNN()
    singlechannelCNN =singlechannelCNN.cuda()

    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()

    learning_rate = 0.01
    optimizer =torch.optim.SGD(singlechannelCNN.parameters(),lr=learning_rate)

    train_data = MyData('single','train')
    test_data = MyData('single','test')
    train_dataloader = DataLoader(train_data,batch_size=20,shuffle=True,drop_last=True,num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=20, shuffle=True, drop_last=True, num_workers=0)
    train(singlechannelCNN, train_dataloader, test_dataloader, epoch, loss_fn, optimizer, Issave, saveEpoch, writepoch)

def TwoChannelCNNtain(epoch, Issave = False, saveEpoch = 10, writepoch = 1):

    twolechannelCNN = TwoChannelCNN()
    twolechannelCNN  =twolechannelCNN .cuda()

    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.cuda()

    learning_rate = 0.01
    optimizer =torch.optim.SGD(twolechannelCNN .parameters(),lr=learning_rate)

    train_data = MyData('two','train')
    test_data = MyData('two','test')
    train_dataloader = DataLoader(train_data,batch_size=20,shuffle=True,drop_last=True,num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=20, shuffle=True, drop_last=True, num_workers=0)
    train(twolechannelCNN , train_dataloader, test_dataloader, epoch, loss_fn,optimizer, Issave, saveEpoch, writepoch)



#returnType：
#print：print the verify result
#getACC：return the verify accuracy
#getTime： return the verify running time
def verify(nnType,widthFile,AllDataSet = True,Datasize = 1000, returnType = 'print'):
    verify_cheat_dir_path = 'dataset/ verify/cheat'
    verify_dir_path = 'dataset/ verify/normal'
    train_cheat_dir_path = 'dataset/train/cheat'
    train_dir_path = 'dataset/train/normal'
    test_cheat_dir_path = 'dataset/test/cheat'
    test_dir_path = 'dataset/test/normal'

    dataset =[]

    if nnType =='single':
        newshape = [1,1,40,40]
    elif nnType =='two':
        newshape = [1, 2, 30, 30]

    if AllDataSet:
        train_cheat_data,train_cheat_fileName = creatDateSet(train_cheat_dir_path,nnType)
        train_normal_data,train_normal_fileName = creatDateSet(train_dir_path,nnType)
        test_cheat_data,test_cheat_fileName = creatDateSet(test_cheat_dir_path,nnType)
        test_normal_data,test_normal_fileName = creatDateSet(test_dir_path, nnType)

        for i in range(len(train_cheat_data)):
            dataset.append([1, torch.reshape(train_cheat_data[i], newshape),train_cheat_fileName[i]])
        for i in range(len(test_cheat_data)):
            dataset.append([1, torch.reshape(test_cheat_data[i], newshape),test_cheat_fileName[i]])

        for i in range(len(train_normal_data)):
            dataset.append([0, torch.reshape(train_normal_data[i], newshape), train_normal_fileName[i]])
        for i in range(len(test_normal_data)):
            dataset.append([0, torch.reshape(test_normal_data[i], newshape), test_normal_fileName[i]])



    verify_cheat_data, verify_cheat_fileName = creatDateSet(verify_cheat_dir_path, nnType)
    verify_normal_data,verify_normal_fileName= creatDateSet(verify_dir_path, nnType)

    for i in range(len(verify_cheat_data)):
        dataset.append([1, torch.reshape(verify_cheat_data[i], newshape), verify_cheat_fileName[i]])

    for i in range(len(verify_normal_data)):
        dataset.append([0, torch.reshape(verify_normal_data[i], newshape), verify_normal_fileName[i]])


    currtNum = 0

    random.shuffle(dataset)
    model = torch.load(widthFile)
    model = model.cuda()
    cheatNum = 0
    normalNum = 0
    wrongList= []
    time_start = time.time()
    for i in range(Datasize):
        label , data, fileName = dataset[i]
        if int(label) == 1:
            cheatNum= cheatNum+1
        elif int(label) == 0:
            normalNum = normalNum+1

        output = model(data.cuda())
        output =output.argmax(1).item()

        if output == int(label):
            currtNum = currtNum +1
        else:
            if returnType == 'print':

                print('{}:label = {},output = {},it\'s worng'.format(fileName,label,output))


    time_end = time.time()
    time_sum = time_end - time_start
    if returnType == 'print':
        print('')
        print('Used model: {}'.format(widthFile))
        print('')
        print('Verify dataset size:  cheat:{} , normal:{}'.format(cheatNum, normalNum ))
        print('')
        print('Accuracy is {}%'.format(float((currtNum/Datasize)*100)))
        print('')
        if currtNum != Datasize:
            print("Wronged in ",end='')
            for num in wrongList:
                print(num, end=',')
            print('')
    elif returnType == 'getAcc':

        return float((currtNum / Datasize) * 100)

    elif returnType == 'getTime':

        return time_sum

def showDataSetSize():
    verify_cheat_dir_path = 'dataset/ verify/cheat'
    verify_dir_path = 'dataset/ verify/normal'
    train_cheat_dir_path = 'dataset/train/cheat'
    train_dir_path = 'dataset/train/normal'
    test_cheat_dir_path = 'dataset/test/cheat'
    test_dir_path = 'dataset/test/normal'
    print("")
    print("train dataset： cheat:{}, normal:{}".format(getFileNum( train_cheat_dir_path),getFileNum(train_dir_path)))
    print("")
    print("test dataset： cheat:{}, normal:{}".format(getFileNum(test_cheat_dir_path ), getFileNum(test_dir_path)))
    print("")
    print("verify dataset： cheat:{}, normal:{}".format(getFileNum(verify_cheat_dir_path), getFileNum(verify_dir_path)))
    print("")

def drawVerifyACC(epoch = 100):
    SingleAcc = []
    TwoACC = []
    for i in range(epoch):
        SingleAcc.append(verify('single', 'SingleChannelCNN_70.pth', Datasize=1000,returnType='getAcc'))
        print('single:{}'.format(i))

    for i in range(epoch):
        TwoACC.append(verify('two', 'TwoChannelCNN_140.pth', Datasize=1000,returnType='getAcc'))
        print('two:{}'.format(i))

    plt.plot(range(epoch), SingleAcc, color='red', label='SingleACC')
    plt.plot(range(epoch), TwoACC, color='blue', label='TwoACC')
    plt.legend(loc=2)
    plt.show()

def drawVerifyTime(epoch = 100):
    SingleAcc = []
    TwoACC = []
    for i in range(epoch):
        SingleAcc.append(verify('single', 'SingleChannelCNN_70.pth', Datasize=1000,returnType='getTime'))
        print('single:{}'.format(i))

    for i in range(epoch):
        TwoACC.append(verify('two', 'TwoChannelCNN_140.pth', Datasize=1000,returnType='getTime'))
        print('two:{}'.format(i))

    plt.plot(range(epoch), SingleAcc, color='red', label='Singletime')
    plt.plot(range(epoch), TwoACC, color='blue', label='Twotime')
    plt.legend(loc=2)
    plt.show()

def drawWrongFile():
    WrongFile_path ='E:\PythonWorkSpace\CheatDetection\dataset\wrong'
    drawAxis(WrongFile_path)

def main():

    #showDataSetSize()

    #SingleChannelCNNtain(100)

    #TwoChannelCNNtain(150)

    #verify('single', 'SingleChannelCNN_70.pth', Datasize=10,returnType='getTime')

    #verify('two', 'TwoChannelCNN_140.pth', Datasize=1000)

    #drawVerifyACC()

    #drawWrongFile()

    pass



if __name__=='__main__':
    main()