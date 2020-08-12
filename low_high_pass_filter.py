# -*- coding: utf-8 -*-  
  
  
from scipy import signal  
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib  
import math  
  
N = 500  #这个是采样的点数，规定时间是1s，那么采样周期为500
fs = 5  
n = [2*math.pi*fs*t/N for t in range(N)]  # n相当于sin的自变量值
axis_x = np.linspace(0,1,num=N)  #时间轴是1s
#设置字体文件，否则不能显示中文  
#myfont = matplotlib.font_manager.FontProperties(fname='c:\\windows\\fonts\\simkai.ttc') #楷体 

myfont = matplotlib.font_manager.FontProperties(fname='c:\\windows\\fonts\\simsun.ttc') #新宋体
  
#频率为5Hz的正弦信号  
x = [math.sin(i) for i in n]  # n不是整数也是可行的，i相当于n中的变量，并不一定是整数
plt.subplot(321)  
plt.plot(axis_x,x)  
plt.title(u'5Hz的正弦信号', fontproperties=myfont)  
plt.axis('tight')  
  
xx = []  #这是一个叠加信号，通过追加方式形成总的叠加信号
x1 = [math.sin(i*10) for i in n]  
for i in range(len(x)):  
    xx.append(x[i] + x1[i])  
   
plt.subplot(322)  
plt.plot(axis_x,xx)  
plt.title(u'5Hz与50Hz的正弦叠加信号', fontproperties=myfont)  
plt.axis('tight')  

b,a = signal.butter(3,0.08,'low')  #求butterworth滤波器的系数  20/500/2=0.08 ,20Hz是低通截止频率
sf = signal.filtfilt(b,a,xx)  #使用滤波器对 xx叠加信号进行滤波
  
plt.subplot(325)  
plt.plot(axis_x,sf)  
plt.title(u'低通滤波后', fontproperties=myfont)  
plt.axis('tight')  
  
b,a = signal.butter(3,0.10,'high')  #求butterworth滤波器的系数  25/500/2=0.1 ,25Hz高通是起始频率
sf = signal.filtfilt(b,a,xx)  
  
plt.subplot(326)  
plt.plot(axis_x,sf)  
plt.title(u'高通滤波后', fontproperties=myfont)  
plt.axis('tight')  
plt.show()

