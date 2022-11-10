import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-1, 1, 100)
y1 = 2 * x + 1
y2 = x ** 2





#增加标注
L1, = plt.plot(x, y1, color='red', linestyle='--', linewidth=1.0)
L2, = plt.plot(x, y2,color='blue',linestyle = '-',linewidth =5.0)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data', 0))
x0 = 0.5
y0 = 2*0.5+1
#画点
plt.scatter(x0,y0,s = 50 ,color ='b')
#画线
plt.plot([x0,x0],[y0,0],'k--')#黑色虚线，（x0，y0），到（x0,0）

plt.show()