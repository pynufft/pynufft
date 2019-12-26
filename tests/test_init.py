from pynufft import NUFFT_cpu, NUFFT_hsa, NUFFT_hsa_legacy, NUFFT
import pynufft
import numpy
dtype = numpy.complex64

def test_init():
    
#     cm = matplotlib.cm.gray
    # load example image
    import pkg_resources
    
    DATA_PATH = pkg_resources.resource_filename('pynufft', 'src/data/')
#     PHANTOM_FILE = pkg_resources.resource_filename('pynufft', 'data/phantom_256_256.txt')
    import numpy
    
#     import matplotlib.pyplot
    
    import scipy

    image = scipy.misc.ascent()[::2,::2]
    image=image.astype(numpy.float)/numpy.max(image[...])

    Nd = (256, 256)  # image space size
    Kd = (512, 512)  # k-space size
    Jd = (6,6)  # interpolation size

    # load k-space points
    om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']

    nfft = NUFFT()  # CPU
#     print(nfft.processor)
    
    nfft.plan(om, Nd, Kd, Jd)
    y = nfft.forward(image)
    x2 = nfft.adjoint(y)
    x3 = nfft.selfadjoint(image)
    print('error=', numpy.linalg.norm(x3 - x2)/numpy.linalg.norm(x2))
    
    device_list = pynufft.helper.device_list()
    
    NufftObj = NUFFT(device_list[1])
    NufftObj._set_wavefront_device(8)
    print('device name = ', NufftObj.device)

    NufftObj.plan(om, Nd, Kd, Jd)
    gx = NufftObj.to_device(image)
    import time
    t0 = time.time()
#     k = nfft.xx2k(nfft.x2xx(image))
    for pp in range(0,50):
#         y = nfft.k2y(nfft.xx2k(nfft.x2xx(image)))    
            y = nfft.forward(image)
#             y = nfft.k2y(k)
#                 k = nfft.y2k(y)
#             x = nfft.adjoint(y)
#             y = nfft.forward(image)
#     y2 = NufftObj.y.get(   NufftObj.queue, async=False)
    t_cpu = (time.time() - t0)/50.0 
    print(t_cpu)
    
#     del nfft
        
#     gy2=NufftObj.forward(gx)
#     gk =     NufftObj.xx2k(NufftObj.x2xx(gx))
    t0= time.time()
    for pp in range(0,50):
#         pass
        gy = NufftObj._forward_device(gx)
#         gy2 = NufftObj.k2y(gk)
#             gx2 = NufftObj.adjoint(gy2)
#             gk2 = NufftObj.y2k(gy2)
#         del gy2
#     c = gx2.get()
#         gy=NufftObj.forward(gx)        
        
    NufftObj.thr.synchronize()
    t_cl = (time.time() - t0)/50
    print(t_cl)
    
    print('gy close? = ', numpy.allclose(y, gy.get(),  atol=numpy.linalg.norm(y)*1e-3))
    print("acceleration=", t_cpu/t_cl)
    maxiter =100
    import time
    t0= time.time()
    x2 =  nfft.solve(y, 'cg',maxiter=maxiter)
#     x2 =  nfft.solve(y, 'L1TVOLS',maxiter=maxiter, rho = 2)
    t1 = time.time()-t0 
#     gy=NufftObj.thr.copy_array(NufftObj.thr.to_device(y2))
    
    t0= time.time()

    x = NufftObj._solve_host(y,'cg', maxiter=maxiter)
#     x = NufftObj.solve(y,'L1TVOLS', maxiter=maxiter, rho=2)
    
    t2 = time.time() - t0
    print(t1, t2)
    print('acceleration=', t1/t2 )
#     k = x.get()
#     x = nfft.k2xx(k)/nfft.st['sn']
#     return
    try:
        import matplotlib.pyplot
        matplotlib.pyplot.subplot(1, 2, 1)
        matplotlib.pyplot.imshow( x.real, cmap= matplotlib.cm.gray, vmin = 0, vmax = 1)
        matplotlib.pyplot.title("HSA reconstruction")
        matplotlib.pyplot.subplot(1, 2,2)
        matplotlib.pyplot.imshow(x2.real, cmap= matplotlib.cm.gray)
        matplotlib.pyplot.title("CPU reconstruction")
        matplotlib.pyplot.show()
#         matplotlib.pyplot.show(block = False)
#         matplotlib.pyplot.pause(3)
#         matplotlib.pyplot.close()
#         del NufftObj.thr
#         del NufftObj
    except:
        print("no graphics")
        
if __name__ == '__main__':
    test_init()    
