import numpy
from pynufft import NUFFT_hsa
import scipy.misc
import matplotlib

Nd = (256,256)
Kd = (512,512)
Jd = (6,6)
om = numpy.random.randn(65536, 2) 
x = scipy.misc.imresize(scipy.misc.ascent(), Nd)
om1 = om[om[:,0]>0, :]
om2 = om[om[:,0]<=0, :]



NufftObj1 = NUFFT_hsa('ocl')
NufftObj1.plan(om1, Nd, Kd, Jd)

NufftObj2 = NUFFT_hsa('cuda')
NufftObj2.plan(om2, Nd, Kd, Jd)

y1 = NufftObj1.forward(x)
y2 = NufftObj2.forward(x)



