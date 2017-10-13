import numpy 
import matplotlib.pyplot as pyplot

try:
    '''
    pip install pynufft
    '''
    from pynufft.pynufft import pynufft
except:
    print('warning: pynufft  not found in system library')
    print('Try to import the local pynufft now')
    import sys
    sys.path.append('../pynufft')
    from pynufft import pynufft
    



om = numpy.random.randn(1512,1)
# print(om)
# print(om.shape)
# pyplot.hist(om)
# pyplot.show()

Nd = (256,) # time grid, tuple
Kd = (512,) # frequency grid, tuple
Jd = (7,) # interpolator 

NufftObj = pynufft()


NufftObj.plan(om, Nd, Kd, Jd)


time_data = numpy.zeros(256, )
time_data[64:192] = 1.0
pyplot.plot(time_data)
pyplot.ylim(-1,2)
pyplot.show()


nufft_freq_data =NufftObj.forward(time_data)

grid_freq_data = NufftObj.y2k_DC(nufft_freq_data)

pyplot.plot(numpy.real(grid_freq_data))  

pyplot.legend('The spectrum with density compensation')
grid_freq_data = NufftObj.y2k(nufft_freq_data)

pyplot.plot(numpy.real(grid_freq_data),'r')
pyplot.legend('The spectrum without density compensation')

pyplot.plot(numpy.real(numpy.fft.fft(numpy.fft.fftshift(time_data),512)),'k:')
pyplot.legend('FFT')


pyplot.show()

im1,=pyplot.plot(numpy.abs(time_data),'r',label='original')
# pyplot.legend(('Original',))

im2,=pyplot.plot(numpy.abs(NufftObj.adjoint(nufft_freq_data)),'b',label='adjoint')

# pyplot.legend(('adjoint',))

im3,=pyplot.plot(numpy.abs(NufftObj.inverse_DC(nufft_freq_data)),'k',label='density-compensated')
pyplot.legend([im1,im2,im3])


pyplot.show()


