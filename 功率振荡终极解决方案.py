"""
test1
实现傅里叶滤波算法，针对一个混杂信号，先使用FFT计算各个频段的幅值，然后挑选想要频段的信号，
其他频段的信号幅值置零，因此形成滤波。然后将信号反FFT得到滤波后的信号。

N是采样信号的周期，采样频率为fs。进行DFT时0~k个点对应的频率是k/N*fs。当N很大时，fs能够
区分的频率也很小，但是不能区分比fs大的频率。
对于采样的N个信号，根据采样频率fs，由DFT可以得到的对应点k时的频率k/N*fs,比如采样频率为25Hz
想要1Hz以下的信号的点滤除，则k的范围是k1/N*fs<f1<f2<k2/N*fs，则k2>N*f2/fs  k1<N*f1/fs

@计算励磁提供的转矩作用
(1) 标幺化 P/SN  Q/SN  I/IN   U/UN
(2) δ=arctan(P/(Q+U*U/Xq))   θ=arctan(Q/P)  发生振荡时，该公式是否成立，或者是有条件的成立
只有其中的全部为相量时，该公式才成立。但振荡发生时，该公式中的量并不全为相量，而是相量之上还有 
在此均为相量实际值，由于默认PMU计算准确，则相量实际值应该是准确的，因此可以实时测算功角和功率因数角
(3) Id=I*sin(δ+θ)
(4) Eqp = U*cos(δ)+Id*Xdp 
(5) FFT滤波求 ΔEqp   Δw  ΔPe    
求出来的量仍然是一个变化量，进一步需要对这个变化量相量化。
需要注意的是对Δw采用(f-50)/50进行滤波，
(6) ∑ΔEqp*Δw     （<0 励磁负阻尼）

@计算原动机转矩的作用
(7)求ΔΔw，由于点很多，可以将起始点置为0，后续的点就按照ΔΔw(i)=Δw(i)-Δw(i-1)
(8)Tj*(ΔΔw)/Δt
(9)ΔPm=Tj*(ΔΔw)/Δt+ΔPe
(10)∑ΔPm*Δw   （>0  原动机负阻尼）

@关于转子转动方程的推算
wm0转子额定机械角速度 wm0=2*pi*f=2*pi*n/60
wm 转子实际机械角速度 
w  标幺化的转子机械角速度，也等同于标幺化的转子电角速度
t时间，单位为s
P为极对数
Tj = J*wm0*wm0/SN
P = T*w

J dwm/dt = Tm - Te
两边乘以wm0*wm0/(SN*wm0)
wm0*wm0/(SN*wm0)*J*dwm/dt = wm0*wm0/(SN*wm0)*(Tm - Te)
J*wm0*wm0/SN*d(wm/wm0)/dt =(Tm - Te)*wm0/SN
Tj*d(wm/wm0)/dt =(Tm - Te)*wm0/SN
标幺化后
Tj*dw/dt =Pm-Pe


test2
实现各种滤波，低通，高通，带通滤波。得到想要的频段信号。

test3
将工况信号标幺化。
计算(1)∫Δw*ΔEq'dt
计算(2)Δw和ΔEq'的过零点得到相位信息

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,ifft
from matplotlib.pylab import mpl
import pandas as pd
import os


#mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
#mpl.rcParams['axes.unicode_minus']=False       #显示负号

#(1) 标幺化 P/SN  Q/SN  I/IN   U/UN
SN=300*10**6/0.9
UN=18*10**3/np.sqrt(3)
IN=SN/UN/3
Xq=0.7
Xdp=0.3
Tj= 8.7

df_meas = pd.read_csv((os.getcwd()).replace("\\","/")+'/hmf0406.csv')
# df_meas = pd.read_csv((os.getcwd()).replace("\\","/")+'/yx1125.csv')
t=df_meas["t"]
Pe=df_meas["P"]/SN*10**6
Qe=df_meas["Q"]/SN*10**6
I=df_meas["IA"]/IN
U=df_meas["UA"]/UN*10**3
ΔF=(df_meas["dF"]-50)/50  #通过实际采样的频率量可能并非运行在50Hz左右，可在后面对ΔF进行滤波
Δw=ΔF*2*np.pi



#(2) δ=arctan(P/(Q+U*U/Xq))   θ=arctan(Q/P)  使用arctan2(y,x)可以得到-180到180范围的角度
#对于这种情况，需要减去90°就不知道是为什么了
#抽水工况  θ= np.arctan2(P,Q)； Q>0 => sin>0 ，P<0 => cos<0  ;   θ应在第2象限，因此需对arctan2进行修正
#抽水工况  δ= np.arctan2((Q+U*U/Xq),P)； P<0 => sin<0 ， Q>0 => cos>0 ;   δ应在第4象限，因此需对arctan2进行修正

θ= np.arctan2(Qe,Pe) 
θ_degree= 180/np.pi*θ
δ= np.arctan2(Pe,(Qe+U*U/Xq)) 
δ_degree= 180/np.pi*δ


#(3) Id=I*sin(δ+θ)
Id = I * np.sin(δ+θ)
Iq = I * np.cos(δ+θ)

#(4) Eqp = U*cos(δ)+Id*Xdp
Eqp = U*np.cos(δ)+Id*Xdp
#Eqp = P*Xdp/(U*np.sin(δ))

#(5) FFT滤波求 ΔEqp   Δw  ΔPe
Ts=0.04
Fs= 1/Ts
#加窗来限定信号范围
# window_tmin = 333  #test
# window_tmax = 338  #test
window_tmin = 158  #406
window_tmax = 162  #406

# window_tmin = 144  #404
# window_tmax = 150  #404

window_min = int(window_tmin/Ts)
window_max = int(window_tmax/Ts)
temp_Eqp = Eqp[window_min:window_max]
temp_Δw = Δw[window_min:window_max]
temp_Pe = Pe[window_min:window_max]
temp_t = t[window_min:window_max]
N= len(temp_Eqp)

#FFT来滤波
yy_Eqp = fft(np.array(temp_Eqp,dtype=np.complex_))/N             # 快速傅里叶变换，归一化
yy_Eqp_abs = np.abs(yy_Eqp)
yy_Δw = fft(np.array(temp_Δw,dtype=np.complex_))/N               # 快速傅里叶变换，归一化
yy_Δw_abs = np.abs(yy_Δw)
yy_Pe = fft(np.array(temp_Pe,dtype=np.complex_))/N               # 快速傅里叶变换，归一化
yy_Eqp_abs = np.abs(yy_Eqp)
k = np.arange(N)
freqyy= k/N*Fs                   # two sides frequency range

# 对于采样的N个信号，根据采样频率fs，由DFT可以得到的对应点k时的频率k/N*fs,比如采样频率为25Hz
# 想要f1、f2频率之外的信号滤除，则k的范围是k1/N*fs<f1<f2<k2/N*fs，则k2>N*f2/fs or k1<N*f1/fs
# 将频率分为4个区段来滤波
f1=1.1 #406
f2=1.5 #406
# f1=1.3 #404
# f2=1.8 #404
f3=Fs-f2
f4=Fs-f1
for i in range(N):
    if  i<int(N*f1/Fs) :
        yy_Eqp[i]=0
        yy_Δw[i]=0
        yy_Pe[i]=0
    elif i>int(N*f2/Fs) and i<int(N*f3/Fs):
        yy_Eqp[i]=0
        yy_Δw[i]=0
        yy_Pe[i]=0
    elif i>int(N*f4/Fs) :
        yy_Eqp[i]=0
        yy_Δw[i]=0
        yy_Pe[i]=0
iyy_ΔEqp = ifft(yy_Eqp)  #需双边频谱才能完全正确的复原信号，单边信号只是用来看的，不能用于计算。
iyy_Δw = ifft(yy_Δw)
iyy_ΔPe = ifft(yy_Pe)

#(6)电磁能量函数计算
S_EPe=np.zeros(N , dtype=np.complex_)
for i in np.arange(N): #0~N-1
    if i==0:
        S_EPe[i] = iyy_ΔEqp[i]*iyy_Δw[i]
    else :
        S_EPe[i] = iyy_ΔEqp[i]*iyy_Δw[i]+S_EPe[i-1]

#(7)求ΔΔw，由于点很多，可以将起始点置为0，后续的点就按照ΔΔw(i)=Δw(i)-Δw(i-1)
ΔΔw=np.zeros(N , dtype=np.complex_)
for i in np.arange(N): #0~N-1
    if i==0:
        ΔΔw[i] = 0
    else :
        ΔΔw[i] = iyy_Δw[i]-iyy_Δw[i-1]

#(8)Tj*(ΔΔw)/Δt
Tj_α =Tj*(ΔΔw)/Ts

#(9)ΔPm=Tj*(ΔΔw)/Δt+ΔPe
ΔPm=Tj_α+iyy_ΔPe

#(10)原动机能量函数计算
S_EPm=np.zeros(N,dtype=np.complex_)
for i in np.arange(N): #0~N-1
    if i==0:
        S_EPm[i] = ΔPm[i]*iyy_Δw[i]
    else :
        S_EPm[i] = ΔPm[i]*iyy_Δw[i]+S_EPm[i-1]

#画图
fig = plt.figure()
fig.add_subplot(331)
plt.plot(t,Pe,'r-.',t,Qe,'b')
plt.legend(["Pe","Qe"])
plt.xlabel("t (s)")
plt.ylabel("Pe,Qe (p.u.)")
plt.subplot(332)
plt.plot(t,U,'r-.',t,I,'b')
plt.legend(["U","I"])
plt.xlabel("t (s)")
plt.ylabel("U,I (p.u.)")
plt.subplot(333)
plt.plot(t,δ_degree,'r-.',t,θ_degree,'b')
plt.legend(["δ","θ"])
plt.xlabel("t (s)")
plt.ylabel("δ,θ (p.u.)")
plt.subplot(334)
plt.plot(t,np.abs(Eqp),'r-.',t,Δw,"b")
plt.legend(["Eqp ","Δω"])
plt.xlabel("t (s)")
plt.ylabel("Eqp,Δω(p.u.)")

ax1=fig.add_subplot(335)
ax1.plot(temp_t,np.abs(iyy_ΔEqp),"r-.")
ax1.legend(["ΔEqp "],loc=4)
ax1.set_xlabel("t (s)")
ax1.set_ylabel("ΔEqp")

ax2 = ax1.twinx()
ax2.plot(temp_t,np.abs(iyy_Δw),'b')
ax2.legend(["Δω"],loc=1)
ax2.set_xlabel("t (s)")
ax2.set_ylabel("Δω(p.u.)")


ax5=fig.add_subplot(336)
ax5.plot(temp_t,np.abs(iyy_ΔPe),"r-.")
ax5.legend(["ΔPe "],loc=4)
ax5.set_xlabel("t (s)")
ax5.set_ylabel("ΔPe(p.u.)")

ax6 = ax5.twinx()
ax6.plot(temp_t,np.abs(ΔPm),'b')
ax6.legend(["ΔPm"],loc=1)
ax6.set_xlabel("t (s)")
ax6.set_ylabel("ΔPm(p.u.)")

ax7=fig.add_subplot(337)
ax7.plot(temp_t,np.abs(S_EPe),"r-.")
ax7.legend(["∑E_lici"],loc=2)
ax7.set_xlabel("t (s)")
ax7.set_ylabel("∑E_lici (p.u.)")

ax8 = ax7.twinx()
ax8.plot(temp_t,np.abs(S_EPm),'b')
ax8.legend(["∑E_yuandongji"],loc=4)
ax8.set_xlabel("t (s)")
ax8.set_ylabel("∑E_yuandongji (p.u.)")


ax3=fig.add_subplot(338)
ax3.plot(freqyy,np.abs(yy_Eqp),"r-.")
ax3.legend(["|ΔEqp| "],loc=9)
ax3.set_xlabel("f (Hz)")
ax3.set_ylabel("|ΔEqp|(p.u.)")

ax4 = ax3.twinx()
ax4.plot(freqyy,np.abs(yy_Δw),"b")
ax4.legend(["|Δω|"],loc=10)
ax4.set_xlabel("f (Hz)")
ax4.set_ylabel("|Δω| (p.u.)")

plt.subplot(339)
plt.plot(temp_t,np.abs(Tj_α),"r-")
plt.legend(["Tjdω/dt "])
plt.xlabel("t (s)")
plt.ylabel("Tjdω/dt(p.u.)")

fig.tight_layout()  # 调整整体空白
plt.subplots_adjust(wspace=1.0, hspace=0.4)  # 调整子图间距
plt.show()