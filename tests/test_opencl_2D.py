"""
Explicitly load the NUFFT_hsa to the 'ocl' backend.
"""
from pynufft import NUFFT_cpu, NUFFT_hsa

import numpy


def test_opencl():
    
    import numpy
    import matplotlib.pyplot

    # load example image
    import pkg_resources
    
    ## Define the source of data 
    DATA_PATH = pkg_resources.resource_filename('pynufft', 'src/data/')
#     PHANTOM_FILE = pkg_resources.resource_filename('pynufft', 'data/phantom_256_256.txt')
    import scipy


    image = scipy.misc.ascent()    
    image = scipy.misc.imresize(image, (256,256))
    image=image.astype(numpy.float)/numpy.max(image[...])

    Nd = (256, 256)  # image space size
    Kd = (512, 512)  # k-space size
    Jd = (6, 6)  # interpolation size

    # load k-space points as M * 2 array
    om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']
    
    # Show the shape of om
    print('the shape of om = ', om.shape)


    # initiating NUFFT_cpu object
    nfft = NUFFT_cpu()  # CPU NUFFT class
    
    # Plan the nfft object
    nfft.plan(om, Nd, Kd, Jd)

    # initiating NUFFT_hsa object
    NufftObj = NUFFT_hsa('ocl', 1, 0)

    # Plan the NufftObj (similar to NUFFT_cpu)
    NufftObj.plan(om, Nd, Kd, Jd, batch = None, radix = 2)

    
    import time
    t0 = time.time()
    for pp in range(0,10):
    
            y = nfft.forward(image)
#             x2 = nfft.adjoint(y)

    t_cpu = (time.time() - t0)/10.0 
    
    
    ## Moving image to gpu
    ## gx is an gpu array, dtype = complex64
    gx = NufftObj.to_device(image)  

    t0= time.time()
    for pp in range(0,100):
        gy = NufftObj.forward(gx)
#         gx2 = NufftObj.adjoint(gy)
    t_ocl = (time.time() - t0)/100
    
    print('t_cpu = ', t_cpu)
    print('t_ocl =, ', t_ocl)
    
    print('gy close? = ', numpy.allclose(y, gy.get(),  atol=numpy.linalg.norm(y)*1e-3))
#     print('gx2 close? = ', numpy.allclose(x2, gx2.get(),  atol=numpy.linalg.norm(x2)*1e-3))
#     matplotlib.pyplot.imshow(gx2.get().real)
#     matplotlib.pyplot.show()
    print("acceleration=", t_cpu/t_ocl)
    maxiter =100
    import time
    t0= time.time()
    x_cpu_cg = nfft.solve(  y, 'cg',  maxiter=maxiter)
#     x2 =  nfft.solve(y2, 'L1TVLAD',maxiter=maxiter, rho = 2)
    t1 = time.time()-t0 
#     gy=NufftObj.thr.copy_array(NufftObj.thr.to_device(y2))
    
    t0= time.time()
    x_cuda_cg = NufftObj.solve( gy,  'cg',   maxiter=maxiter)
#     x = NufftObj.solve(gy,'L1TVLAD', maxiter=maxiter, rho=2)
    
    t2 = time.time() - t0
    print(t1, t2)
    print('acceleration of cg=', t1/t2 )


    t0= time.time()
    x_cpu_TV =  nfft.solve(y, 'L1TVOLS',maxiter=maxiter, rho = 2)
    t1 = time.time()-t0 
    
    t0= time.time()
    print(gy.shape)
    x_cuda_TV = NufftObj.solve(gy,'L1TVOLS', maxiter=maxiter, rho=2)
    
    t2 = time.time() - t0
    print(t1, t2)
    print('acceleration of TV=', t1/t2 )
    try:
        matplotlib.pyplot.subplot(2, 2, 1)
        matplotlib.pyplot.imshow( x_cpu_cg.real, cmap= matplotlib.cm.gray)
        matplotlib.pyplot.title('CG_cpu')
        matplotlib.pyplot.subplot(2, 2, 2)
        matplotlib.pyplot.imshow(x_cuda_cg.get().real, cmap= matplotlib.cm.gray)
        matplotlib.pyplot.title('CG_ocl')
        matplotlib.pyplot.subplot(2, 2, 3)
        matplotlib.pyplot.imshow( x_cpu_TV.real, cmap= matplotlib.cm.gray)
        matplotlib.pyplot.title('TV_cpu')
        matplotlib.pyplot.subplot(2, 2, 4)
        matplotlib.pyplot.imshow(x_cuda_TV.get().real, cmap= matplotlib.cm.gray)
        matplotlib.pyplot.title('TV_ocl')    
        matplotlib.pyplot.show()
    except:
        print('no matplotlib')
    
    NufftObj.release()
    del NufftObj
if __name__ == '__main__':
    test_opencl()    
