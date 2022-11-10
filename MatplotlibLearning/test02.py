import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 100)
y1 = 2 * x + 1
y2 = x ** 2





#增加图例
L1, = plt.plot(x, y1, color='red', linestyle='--', linewidth=1.0)
L2, = plt.plot(x, y2,color='blue',linestyle = '-',linewidth =5.0)
plt.legend(handles = [L1,L2],labels = ['test1','test2'],loc ='best')
plt.show()