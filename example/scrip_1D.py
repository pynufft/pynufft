import numpy 
import matplotlib.pyplot as pyplot
    
from pynufft.pynufft import NUFFT_cpu, NUFFT_hsa


om = numpy.random.randn(1512,1)
 
Nd = (256,) # time grid, tuple
Kd = (512,) # frequency grid, tuple
Jd = (7,) # interpolator 

NufftObj = NUFFT_cpu()

NufftObj.plan(om, Nd, Kd, Jd)

time_data = numpy.zeros(256, )
time_data[64:192] = 1.0
pyplot.plot(time_data)
pyplot.ylim(-1,2)
pyplot.show()


nufft_freq_data =NufftObj.forward(time_data)
pyplot.plot(om,nufft_freq_data.real,'.', label='real')
pyplot.plot(om,nufft_freq_data.imag,'r.', label='imag')
pyplot.legend()
pyplot.show()


restore_time = NufftObj.solve(nufft_freq_data,'cg', maxiter=30)
restore_time1 = NufftObj.solve(nufft_freq_data,'L1TVLAD', maxiter=30,rho=1)
restore_time2 = NufftObj.solve(nufft_freq_data,'L1TVOLS', maxiter=30,rho=1)

im1,=pyplot.plot(numpy.abs(time_data),'r',label='original signal')
im2,=pyplot.plot(numpy.abs(restore_time1),'b:',label='L1TVLAD')
im3,=pyplot.plot(numpy.abs(restore_time2),'k--',label='L1TVOLS')
im4,=pyplot.plot(numpy.abs(restore_time),'r:',label='conjugate_gradient_method')
pyplot.legend([im1, im2, im3,im4])
pyplot.show()


