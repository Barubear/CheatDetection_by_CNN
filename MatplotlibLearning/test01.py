import matplotlib.pyplot as plt
import numpy as np


#基础用法
def test01():
    x = np.linspace(-1,1,100)#-1到1生成100个点
    y = 2*x+1
    #plt.plot(x,y)
    #plt.show()
#两条曲线两张图
def test02():
    x = np.linspace(-1, 1, 100)
    y1 = 2*x+1
    y2 = x**2
    #plt.figure()
    #plt.plot(x, y1)

    #plt.figure()
    #plt.plot(x,y2)

    #plt.show()
#两条曲线一张图
def test03():
    x = np.linspace(-1, 1, 100)
    y1 = 2*x+1
    y2 = x**2

    #plt.plot(x, y1,color='red',linestyle = '--',linewidth =1.0)
    #plt.plot(x, y2,color='blue',linestyle = '-',linewidth =5.0)

    #plt.show()
#设置坐标轴
def test04():
    x = np.linspace(-1, 1, 100)
    y1 = 2 * x + 1
    y2 = x ** 2
    #xyf范围
    plt.xlim((-1,2))
    plt.ylim((-2,3))
    #xy轴描述
    plt.xlabel('It\'s x')
    plt.ylabel('It\'s y')
    #xy轴间隔重设
    new_ticks = np.linspace(-2,2,11)
    plt.xticks(new_ticks)
    plt.yticks([-1,0,1,2,3],
               ['lv1','lv2','lv3','lv4','lv5'])
    #设置边框
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    #设置x轴刻度位置为bottom
    #y轴刻度位置为left
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('right')
    # 设置x轴位置到x= 0处
    # y轴刻度位置到y= 0处
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['left'].set_position(('data', 0))

    plt.plot(x, y1, color='red', linestyle='--', linewidth=1.0)
    plt.plot(x, y2,color='blue',linestyle = '-',linewidth =5.0)
    plt.show()

if __name__ == '__main__':
    test04()