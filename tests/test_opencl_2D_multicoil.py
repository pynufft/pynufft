"""
Explicitly load the NUFFT_hsa to the 'ocl' backend.
"""
from pynufft import NUFFT_cpu, NUFFT_hsa_legacy, NUFFT_hsa, NUFFT, helper

import numpy


def test_opencl_multicoils():
    
    import numpy
    import matplotlib.pyplot

    # load example image
    import pkg_resources
    
    ## Define the source of data 
    DATA_PATH = pkg_resources.resource_filename('pynufft', 'src/data/')
#     PHANTOM_FILE = pkg_resources.resource_filename('pynufft', 'data/phantom_256_256.txt')
    import scipy


    image = scipy.misc.ascent()[::2,::2]
    image=image.astype(numpy.float)/numpy.max(image[...])

    Nd = (256, 256)  # image space size
    Kd = (512, 512)  # k-space size
    Jd = (6, 6)  # interpolation size

    # load k-space points as M * 2 array
    om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']
    
    # Show the shape of om
    print('the shape of om = ', om.shape)

    batch = 8

    # initiating NUFFT_cpu object
    nfft = NUFFT()  # CPU NUFFT class
    
    # Plan the nfft object
    nfft.plan(om, Nd, Kd, Jd, batch = batch)
    device_list = helper.device_list()
    # initiating NUFFT_hsa object
#     try:
    NufftObj = NUFFT(device_list[2], legacy=False)
#     NufftObj._set_wavefront_device(4)
    print(NufftObj.thr)
#     except:
#         try:
#             NufftObj = NUFFT_hsa('ocl', 1, 0)
#         except:
#             NufftObj = NUFFT_hsa('ocl', 0, 0)

    # Plan the NufftObj (similar to NUFFT_cpu)
    NufftObj.plan(om, Nd, Kd, Jd, batch= batch)
    coil_sense = numpy.ones(Nd + (batch,), dtype = numpy.complex64)
    for cc in range(0, batch, 2):
        coil_sense[ int(256/batch)*cc:int(256/batch)*(cc+1), : ,cc].real *= 0.1
        coil_sense[:, int(256/batch)*cc:int(256/batch)*(cc+1),cc].imag *= -0.1  
        
    NufftObj.set_sense(coil_sense )
    nfft.set_sense(coil_sense)
    y = nfft.forward_one2many(image)
    import time
    t0 = time.time()
    for pp in range(0,2):
    
            
            xx = nfft.adjoint_many2one(y)

    t_cpu = (time.time() - t0)/2
    
    
    ## Moving image to gpu
    ## gx is an gpu array, dtype = complex64
#     gx = NufftObj.to_device(image)  
    
    gy = NufftObj.forward_one2many(image)
    
    t0= time.time()
    for pp in range(0,10):
        
        gxx = NufftObj.adjoint_many2one(gy)
    t_cu = (time.time() - t0)/10
    print(y.shape, gy.shape)
    print('t_cpu = ', t_cpu)
    print('t_cuda =, ', t_cu)
    
    print('gy close? = ', numpy.allclose(y, gy,  atol=numpy.linalg.norm(y)*1e-6))
    print('gy error = ', numpy.linalg.norm(y- gy)/numpy.linalg.norm(y))
    print('gxx close? = ', numpy.allclose(xx, gxx,  atol=numpy.linalg.norm(xx)*1e-6))
    print('gxx error = ', numpy.linalg.norm(xx- gxx)/numpy.linalg.norm(xx))
#     for bb in range(0, batch):
    matplotlib.pyplot.subplot(1,2,1)
    matplotlib.pyplot.imshow( xx[...].real, cmap= matplotlib.cm.gray)
    matplotlib.pyplot.title('Adjoint_cpu_coil')
    matplotlib.pyplot.subplot(1,2,2)
    matplotlib.pyplot.imshow(gxx[...].real, cmap= matplotlib.cm.gray)
    matplotlib.pyplot.title('Adjoint_hsa_coil')
#         matplotlib.pyplot.subplot(2, 2, 3)
#         matplotlib.pyplot.imshow( x_cpu_TV.real, cmap= matplotlib.cm.gray)
#         matplotlib.pyplot.title('TV_cpu')#     x_cuda_TV = NufftObj.solve(gy,'L1TVOLS', maxiter=maxiter, rho=2)
#         matplotlib.pyplot.subplot(2, 2, 4)
#         matplotlib.pyplot.imshow(x_cuda_TV.get().real, cmap= matplotlib.cm.gray)
#         matplotlib.pyplot.title('TV_cuda')    
    matplotlib.pyplot.show(block=False)
    matplotlib.pyplot.pause(4)
    matplotlib.pyplot.close()
        
    print("acceleration=", t_cpu/t_cu)
    maxiter =100
    import time
    t0= time.time()
    x_cpu_cg = nfft.solve(y, 'cg',maxiter=maxiter)
#     x2 =  nfft.solve(y2, 'L1TVLAD',maxiter=maxiter, rho = 2)
    t1 = time.time()-t0 
#     gy=NufftObj.thr.copy_array(NufftObj.thr.to_device(y2))
    
    t0= time.time()
    x_cuda_cg = NufftObj.solve(gy,'cg', maxiter=maxiter)
#     x = NufftObj.solve(gy,'L1TVLAD', maxiter=maxiter, rho=2)
    print('shape of cg = ', x_cuda_cg.shape, x_cpu_cg.shape)
    t2 = time.time() - t0
    print(t1, t2)
    print('acceleration of cg=', t1/t2 )

    t0= time.time()
#     x_cpu_TV =  nfft.solve(y, 'L1TVOLS',maxiter=maxiter, rho = 2)
    t1 = time.time()-t0 
    
    t0= time.time()
    
#     x_cuda_TV = NufftObj.solve(gy,'L1TVOLS', maxiter=maxiter, rho=2)
    
    t2 = time.time() - t0
    print(t1, t2)
#     print('acceleration of TV=', t1/t2 )
    
#     try:
    for bb in range(0, batch):
        matplotlib.pyplot.subplot(2, batch, 1 + bb)
        matplotlib.pyplot.imshow( x_cpu_cg[...,bb].real, cmap= matplotlib.cm.gray)
        matplotlib.pyplot.title('CG_cpu_coil_' + str(bb))
        matplotlib.pyplot.subplot(2, batch, 1 + batch + bb)
        matplotlib.pyplot.imshow(x_cuda_cg[...,bb].real, cmap= matplotlib.cm.gray)
        matplotlib.pyplot.title('CG_hsa_coil_' + str(bb))
#         matplotlib.pyplot.subplot(2, 2, 3)
#         matplotlib.pyplot.imshow( x_cpu_TV.real, cmap= matplotlib.cm.gray)
#         matplotlib.pyplot.title('TV_cpu')#     x_cuda_TV = NufftObj.solve(gy,'L1TVOLS', maxiter=maxiter, rho=2)
#         matplotlib.pyplot.subplot(2, 2, 4)
#         matplotlib.pyplot.imshow(x_cuda_TV.get().real, cmap= matplotlib.cm.gray)
#         matplotlib.pyplot.title('TV_cuda')    
    matplotlib.pyplot.show()
#     except:
#         print('no matplotlib')
    
    NufftObj.release()
    del NufftObj
if __name__ == '__main__':
    test_opencl_multicoils()
#     test_forward()    
