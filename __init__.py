

# from src._transform.transform_cpu import NUFFT as NUFFT_cpu
# from src._transform.transform_hsa import NUFFT as NUFFT_hsa
# 
# def test_init():
#     import cProfile
#     import numpy
#     import matplotlib.pyplot
#     import copy
# 
#     cm = matplotlib.cm.gray
#     # load example image
#     import pkg_resources
#     
#     DATA_PATH = pkg_resources.resource_filename('pynufft', 'src/data/')
# #     PHANTOM_FILE = pkg_resources.resource_filename('pynufft', 'data/phantom_256_256.txt')
#     import numpy
#     import matplotlib.pyplot
#     import scipy
#     # load example image
# #     image = numpy.loadtxt(DATA_PATH +'phantom_256_256.txt')
# #     image = scipy.misc.face(gray=True)
#     image = scipy.misc.ascent()    
#     image = scipy.misc.imresize(image, (256,256))
#     
#     image=image.astype(numpy.float)/numpy.max(image[...])
#     #numpy.save('phantom_256_256',image)
#     matplotlib.pyplot.subplot(1,3,1)
#     matplotlib.pyplot.imshow(image, cmap=matplotlib.cm.gray)
#     matplotlib.pyplot.title("Load Scipy \"ascent\" image")
# #     matplotlib.pyplot.show()
#     print('loading image...')
# #     image[128, 128] = 1.0
#     Nd = (256, 256)  # image space size
#     Kd = (512, 512)  # k-space size
#     Jd = (6, 6)  # interpolation size
# 
#     # load k-space points
#     om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']
# 
#     # create object
#     
# #         else:
# #             n_shift=tuple(list(n_shift)+numpy.array(Nd)/2)
# #     from transform_cpu import NUFFT as NUFFT_c
#     nfft = NUFFT_cpu()  # CPU
#     nfft.plan(om, Nd, Kd, Jd)
# #     nfft.initialize_gpu()
#     import scipy.sparse
# #     scipy.sparse.save_npz('tests/test.npz', nfft.st['p'])
# 
#     NufftObj = NUFFT_hsa()
# 
#     NufftObj.plan(om, Nd, Kd, Jd)
#     NufftObj.offload(API='ocl',   platform_number= 0 , device_number= 0)
# #     print('sp close? = ', numpy.allclose( nfft.st['p'].data,  NufftObj.st['p'].data, atol=1e-1))
# #     NufftObj.initialize_gpu()
# 
#     y = nfft.k2y(nfft.xx2k(nfft.x2xx(image)))
#     NufftObj.x_Nd = NufftObj.thr.to_device( image.astype(dtype))
#     gx = NufftObj.thr.copy_array(NufftObj.x_Nd)
#     print('x close? = ', numpy.allclose(image, NufftObj.x_Nd.get() , atol=1e-4))
#     NufftObj._x2xx()    
# #     ttt2= NufftObj.thr.copy_array(NufftObj.x_Nd)
#     print('xx close? = ', numpy.allclose(nfft.x2xx(image), NufftObj.x_Nd.get() , atol=1e-4))        
# 
#     NufftObj._xx2k()    
#     
# #     print(NufftObj.k_Kd.get(queue=NufftObj.queue, async=True).flags)
# #     print(nfft.xx2k(nfft.x2xx(image)).flags)
#     k = nfft.xx2k(nfft.x2xx(image))
#     print('k close? = ', numpy.allclose(nfft.xx2k(nfft.x2xx(image)), NufftObj.k_Kd.get() , atol=1e-3*numpy.linalg.norm(k)))   
#     
#     NufftObj._k2y()    
#     
#     
#     NufftObj._y2k()
#     y2 = NufftObj.y.get(   )
#     
#     print('y close? = ', numpy.allclose(y, y2 ,  atol=1e-3*numpy.linalg.norm(y)))
# #     print(numpy.mean(numpy.abs(nfft.y2k(y)-NufftObj.k_Kd2.get(queue=NufftObj.queue, async=False) )))
#     print('k2 close? = ', numpy.allclose(nfft.y2k(y2), NufftObj.k_Kd2.get(), atol=1e-3*numpy.linalg.norm(nfft.y2k(y2)) ))   
#     NufftObj._k2xx()
# #     print('xx close? = ', numpy.allclose(nfft.k2xx(nfft.y2k(y2)), NufftObj.xx_Nd.get(queue=NufftObj.queue, async=False) , atol=0.1))
#     NufftObj._xx2x()
#     print('x close? = ', numpy.allclose(nfft.adjoint(y2), NufftObj.x_Nd.get() , atol=1e-3*numpy.linalg.norm(nfft.adjoint(y2))))
#     image3 = NufftObj.x_Nd.get() 
#     import time
#     t0 = time.time()
#     for pp in range(0,10):
# #         y = nfft.k2y(nfft.xx2k(nfft.x2xx(image)))    
# #             x = nfft.adjoint(y)
#             y = nfft.forward(image)
# #     y2 = NufftObj.y.get(   NufftObj.queue, async=False)
#     t_cpu = (time.time() - t0)/10.0 
#     print(t_cpu)
#     
# #     del nfft
#         
#     
#     t0= time.time()
#     for pp in range(0,100):
# #         pass
# #         NufftObj.adjoint()
#         gy=NufftObj.forward(gx)        
#         
# #     NufftObj.thr.synchronize()
#     t_cl = (time.time() - t0)/100
#     print(t_cl)
#     print('gy close? = ', numpy.allclose(y,gy.get(),  atol=1e-1))
#     print("acceleration=", t_cpu/t_cl)
#     maxiter =100
#     import time
#     t0= time.time()
# #     x2 = nfft.solver(y2, 'cg',maxiter=maxiter)
#     x2 =  nfft.solver(y2, 'L1TVLAD',maxiter=maxiter, rho = 2)
#     t1 = time.time()-t0
# #     gy=NufftObj.thr.copy_array(NufftObj.thr.to_device(y2))
#     
#     t0= time.time()
# #     x = NufftObj.solver(gy,'dc', maxiter=maxiter)
#     x = NufftObj.solver(gy,'L1TVLAD', maxiter=maxiter, rho=2)
#     
#     t2 = time.time() - t0
#     print(t1, t2)
#     print('acceleration=', t1/t2 )
# #     k = x.get()
# #     x = nfft.k2xx(k)/nfft.st['sn']
# #     return
#     
#     matplotlib.pyplot.subplot(1, 3, 2)
#     matplotlib.pyplot.imshow( x.get().real, cmap= matplotlib.cm.gray)
#     matplotlib.pyplot.subplot(1, 3,3)
#     matplotlib.pyplot.imshow(x2.real, cmap= matplotlib.cm.gray)
#     matplotlib.pyplot.show()