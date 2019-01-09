import numpy 
import matplotlib.pyplot as pyplot

# try:
#     '''
#     pip install pynufft
#     '''
#     from pynufft.pynufft import pynufft
# except:
#     print('warning: pynufft  not found in system library')
#     print('Try to import the local pynufft now')
#     import sys
#     sys.path.append('../pynufft')
#     from pynufft import pynufft
    

def example_1D():

    om = numpy.random.randn(1512,1)
    # print(om)
    # print(om.shape)
    # pyplot.hist(om)
    # pyplot.show()
    
    Nd = (256,) # time grid, tuple
    Kd = (512,) # frequency grid, tuple
    Jd = (7,) # interpolator 
    from pynufft import NUFFT_cpu, NUFFT_hsa
    NufftObj = NUFFT_cpu()
    
    batch = 4
    
    NufftObj.plan(om, Nd, Kd, Jd, batch = batch)
    
    
    
    time_data = numpy.zeros( (256, batch) )
    time_data[64:192,:] = 1.0
    pyplot.plot(time_data)
    pyplot.ylim(-1,2)
    pyplot.show()
    
    
    nufft_freq_data =NufftObj.forward(time_data)
    print('shape of y = ', nufft_freq_data.shape)
    
    x2 =NufftObj.adjoint(nufft_freq_data)
    restore_time = NufftObj.solve(nufft_freq_data,'cg', maxiter=30)
    
    restore_time1 = NufftObj.solve(nufft_freq_data,'L1TVOLS', maxiter=30,rho=1)
# 
#     restore_time2 = NufftObj.solve(nufft_freq_data,'L1TVOLS', maxiter=30,rho=1)
#     
#     im1,=pyplot.plot(numpy.abs(time_data),'r',label='original signal')
 
    
#     im3,=pyplot.plot(numpy.abs(restore_time2),'k--',label='L1TVOLS')
#     im4,=pyplot.plot(numpy.abs(restore_time),'r:',label='conjugate_gradient_method')
#     pyplot.legend([im1, im2, im3,im4])
    
    
    for slice in range(0, batch):
        pyplot.plot(numpy.abs(x2[:,slice]))
        pyplot.plot(numpy.abs(restore_time[:,slice]))
        pyplot.show()

if __name__ == '__main__':
    example_1D()
