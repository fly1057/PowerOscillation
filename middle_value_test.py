# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 19:53:27 2017

@author: ll
"""

#中值算法

import numpy as np
import scipy as sp
#import scipy.signal as signal

a=np.random.randint(1,100,100)
a=list(a)
print a 
b=a[:]    #如果希望a和b没有关系就用索引方法，而不用引用方法
b.sort()
b_middle=b[len(b)/2]
b_middle2=sp.signal.medfilt(b,5)
print b_middle
print b_middle2
print b
print a 