# -*- coding: utf-8 -*-
"""
Created on Mon Oct 09 16:47:55 2017

@author: ll
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:27:45 2017

@author: ll
"""
from os import path
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal  as signal
import pandas as pd
#from scipy.fftpack import fft, rfft


#1初始化数据.................................................

path_str=u"C:/Users/ll/Desktop/wmp2015-06-02@02-21-10.211600_.csv"
dir_filename,filetype=path.splitext(path_str)   
df=pd.read_csv(path_str,encoding="gb2312")
nrow,ncolumn=df.shape  #实际上列数包含了索引


#将CSV数据的表头进行更换，便于编程处理
df.columns = [u'point_number',u't',u'Ua',u'f',u'Pe',u'Qe', u'delta_f',u'delta_Pe'] 
df=df.drop([u'point_number'],axis=1)
df.t=np.arange(nrow)*0.0002
t0=df.t
P0=df.Pe
Q0=df.Qe
Ut0=df.Ua
delta_f0=df.delta_f



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
UN=10000
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
#[b,a]=signal.butter(3,[0.002,0.01],'bandpass')
#P_filt = signal.filtfilt(b,a,P)
#plt.plot(t,P_filt,linestyle=':')

#[b,a]=signal.butter(3,[0.002,0.01],'bandpass')
#w_filt = signal.filtfilt(b,a,delta_w)
#plt.plot(t,delta_w,linestyle='-')



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
plt.plot(t,P,linestyle='-')
plt.plot(t,delta_Eqp,linestyle='-')
plt.plot(t,Eqp,linestyle='-')
plt.plot(t,delta_w,linestyle='-')
plt.plot(t,S,linestyle='-')
plt.plot(t,Q,linestyle='-.')
plt.plot(t,Ut,linestyle=':')
plt.legend(['P','Q','Ut'])

plt.grid()
plt.xlabel("t /s")
plt.ylabel("p.u.")
#plt.ylim([1.025,1.04])
plt.xlim([0,100])


