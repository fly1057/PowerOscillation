# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:27:45 2017

@author: ll
"""

import numpy as np
#from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import scipy.signal  as signal
#from scipy.fftpack import fft, rfft


#1初始化数据.................................................
#提取数据
t0,P0,Q0,delta_f0,Ut0=np.loadtxt('./testdata_wmp.csv', delimiter=',', usecols=(1,5,6,7,3), unpack=True) 

t0=np.arange(np.size(t0))*0.0002



#选定考察的数据的 起始-终止点
gaps=0.0002   #对应以秒为单位，时间间隔的数值
start_time=0
end_time=10
start_point=int(start_time/gaps)  #强迫类型转换,如果不这样则list将无法进行索引
end_point=int(end_time/gaps)      #强迫类型转换

delta_time=end_time-start_time
delta_point=end_point-start_point
Fs=delta_point/delta_time
Fa=Fs/2

#按照起始-终止点进行筛选
t=(np.arange(np.size(t0)))[start_point:end_point]*0.01
P=P0[start_point:end_point]
Q=Q0[start_point:end_point]
delta_f=delta_f0[start_point:end_point]
Ut=Ut0[start_point:end_point]

#标幺化
#1号机
SN=90
UN=100
Q=Q/SN
P=P/SN
Ut=Ut/UN
Xdp=0.1965
Xq=1.836
delta_w=2*np.pi*delta_f


#中值滤波
#plt.plot(t,P)
#P=sp.signal.medfilt(P)
#plt.plot(t,P)

#fft计算
#P_fft=abs(fft(P))
#
#plt.plot(t,P_fft,linestyle='-')

# Butterworth滤波
# 采样点数8000，时间80s，采样频率FS=100，需要得到的频率为3Hz，使用带通滤波

#如果使用wn=0.1*2/100~5*2/100=0.002~0.1 ， 就实现不了
#[b,a]=signal.butter(3,[0.002,0.1],'bandpass')
#P_filt = signal.filtfilt(b,a,P)
#plt.plot(t,P_filt,linestyle='-')

#但是使用wn=0.1*2/100~0.5*2/100=0.002~0.01 ， 就能实现,不管怎样按照这个
#滤波器进行滤波就能实现带通滤波器
[b,a]=signal.butter(3,[0.002,0.01],'bandpass')
P_filt = signal.filtfilt(b,a,P)
plt.plot(t,P_filt,linestyle=':')

[b,a]=signal.butter(3,[0.002,0.01],'bandpass')
w_filt = signal.filtfilt(b,a,delta_w)
plt.plot(t,delta_w,linestyle='-')



#求功角δ   tan δp=P/(Q+Ut^2/Xd’)
bp=np.arctan(P/(Q+Ut*Ut/Xdp))
bp_degree=bp/np.pi*180

#求Eqp  Eq’=P*Xd’/(Ut*sinδ)
Eqp=P*Xdp/(Ut*np.sin (bp))
Eqp_average=np.average(Eqp)

#求ΔEqp
delta_Eqp=Eqp-Eqp_average

#求Δω

#求∫ΔEqp*Δω
S=delta_Eqp*delta_w  #首先形成一个一维数组，每个元素为对应数组点的乘积
S_len=np.size(S)
for i in np.arange(S_len):
    if i==0:
        Sum_temp=S[i]
        S[i]=Sum_temp
    else :
        Sum_temp=S[i]+Sum_temp   #求积分实际上是求和     
        S[i]=Sum_temp 

#画图
#plt.plot(t,P,linestyle='-')
#plt.plot(t,delta_Eqp,linestyle='-')
#plt.plot(t,Eqp,linestyle='-')
#plt.plot(t,delta_w,linestyle='-')
#plt.plot(t,S,linestyle='-')
#plt.plot(t,Q,linestyle='-.')
#plt.plot(t,Ut,linestyle=':')
#plt.legend(['P','Q','Ut'])

###设定图像大小，像素，去除边框
#plt.figure(figsize=(7,4), dpi=100)
#ax = plt.gca()
#ax.spines['top'].set_visible(False)  #去掉上边框
#ax.spines['right'].set_visible(False) #去掉右边框

plt.grid()
plt.xlabel("t /s")
plt.ylabel("p.u.")
#plt.ylim([1.025,1.04])
plt.xlim([0,100])
plt.show()


