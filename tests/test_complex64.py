print('hello world')
import pynufft
from pynufft import NUFFT, NUDFT_cpu
import numpy
import finufftpy
om = numpy.array(numpy.pi*2*(numpy.random.rand(int(1e+6)) - 0.5),ndmin=2).T

print('om.shape=',om.shape)

import time


Nd = (256, )
Kd = (512, )
Jd= (7, )
device_flag = pynufft.helper.device_list()[0]
F1 = NUFFT(device_flag, legacy=True)
# device_flag = pynufft.helper.device_list()[0]
# F1 = NUFFT()
F1.plan(om, Nd, Kd, Jd)

F2 = NUDFT_cpu()
F2.plan(om, Nd)

x = numpy.random.randn(*Nd)
# gx = F1.to_device(x)
t0=time.time()
for pp in range(0,100):
    y1 = F1.forward(x)
#     gy1 = F1._forward_legacy(gx)
t1=time.time() 
for pp in range(0,1):
    y2 = F2.forward(x)
t2 = time.time()
# y1 = gy1.get()
print('NUFFT=', (t1-t0)/100)
print('NUDFT=', (t2-t1)/1)
# print(y1.shape)
x1 = F1.adjoint(y2)
x2 = F2.adjoint(y2)/2
# print(y1[0], y2[0])
print(numpy.linalg.norm(y1 - y2)/numpy.linalg.norm(y2))
# print(numpy.linalg.norm(x1 - x2)/numpy.linalg.norm(x2))
# import matplotlib.pyplot

cj=numpy.zeros(om.shape,dtype=numpy.complex128); # output
# fk=np.random.rand(ms)+1j*np.random.rand(ms);
eps=1e-6

timer=time.time()  
for pp in range(0,100):
    ret=finufftpy.nufft1d2(om,cj,1,eps,x)
elapsed=time.time()-timer

print('finufftpy=', elapsed/100)
print(numpy.linalg.norm(cj.conj().T - y2)/numpy.linalg.norm(y2))
# print(cj.conj().T - y2)
# import matplotlib.pyplot
# matplotlib.pyplot.plot(cj)
# matplotlib.pyplot.show()
# matplotlib.pyplot.plot(om)
# matplotlib.pyplot.show()