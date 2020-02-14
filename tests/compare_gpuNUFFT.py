from pynufft import NUFFT_cpu, NUFFT_hsa

import numpy
dtype = numpy.complex64

def test_init(batch):
    print('batch=', batch)
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
    radial_path = '/home/sram/Cambridge_2012/WORD_PPTS/multicoil_NUFFT/gpuNUFFT/matlab/demo/data/brain_64spokes.mat'
    
    # load k-space points
#     om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']
    import scipy.io
    k = scipy.io.loadmat('/home/sram/Cambridge_2012/WORD_PPTS/multicoil_NUFFT/gpuNUFFT/matlab/demo/data/brain_64spokes.mat')['k'].flatten(order='F')*numpy.pi
    rawdata = scipy.io.loadmat('/home/sram/Cambridge_2012/WORD_PPTS/multicoil_NUFFT/gpuNUFFT/matlab/demo/data/brain_64spokes.mat')['rawdata']
#     print('rawdata = ', rawdata.shape)
    om = numpy.zeros((512*64, 2), dtype = numpy.float64)
    om[:,0] = k.real
    om[:,1] = k.imag 
    nfft = NUFFT_cpu()  # CPU
#     batch = 32
    nfft.plan(om, Nd, Kd, Jd, batch=batch)
#     try:
#     NufftObj = NUFFT_hsa(   'cuda',    0,  0)
#     NufftObj = NUFFT_hsa(   'cuda',    0,  0)

    NufftObj = NUFFT_hsa('ocl',1,0)
#     NufftObj2 = NUFFT_hsa('cuda',0,0)
    
    NufftObj.plan(om, Nd, Kd, Jd, radix=1, batch = batch)
    x0 = numpy.reshape(image, (256,256,1))
    x = numpy.broadcast_to(x0, (256,256, batch))
#     print(x.shape)
    y = NufftObj.forward(x).get()
#     y = nfft.forward(x)
    
    import time
    t0 = time.time()
# #     k = nfft.xx2k(nfft.x2xx(image))
    for pp in range(0,10):
# #         y = nfft.k2y(nfft.xx2k(nfft.x2xx(image)))    
# #             y = nfft.forward(image)
# #             y = nfft.k2y(k)
# #                 k = nfft.y2k(y)
            x2 = nfft.adjoint(y)
# #             y = nfft.forward(image)
# #     y2 = NufftObj.y.get(   NufftObj.queue, async=False)
    t_cpu_adj = (time.time() - t0)/10 

    
#     del nfft
        
    gy2=NufftObj.to_device(y)
#     gk =     NufftObj.xx2k(NufftObj.x2xx(gx))
    t0= time.time()
    for pp in range(0,100):

            gx2 = NufftObj.adjoint(y)
            gx2.get()
#             gx2 = NufftObj.adjoint(gy2)

    t_gpu_adj_transfer = (time.time() - t0)/100

    t0= time.time()
    for pp in range(0,100):

            gx2 = NufftObj.adjoint(gy2)
    NufftObj.thr.synchronize()
    t_gpu_adj_no_transfer = (time.time() - t0)/100
    
    print('gx2 close? = ', numpy.allclose(x2, gx2.get(),  atol=numpy.linalg.norm(x2)*1e-6))
    
    
    
    t0 = time.time()
# #     k = nfft.xx2k(nfft.x2xx(image))
    for pp in range(0,10):
# #         y = nfft.k2y(nfft.xx2k(nfft.x2xx(image)))    
            y = nfft.forward(x)
# #             y = nfft.k2y(k)
# #                 k = nfft.y2k(y)
# #             x = nfft.adjoint(y)
# #             y = nfft.forward(image)
# #     y2 = NufftObj.y.get(   NufftObj.queue, async=False)
    t_cpu_forward = (time.time() - t0)/10.0 
    
    
#     del nfft
        
#     gy2=NufftObj.forward(gx)
#     gk =     NufftObj.xx2k(NufftObj.x2xx(gx))
    gx = NufftObj.to_device(x)
    t0= time.time()
    for pp in range(0,100):
#         pass
#             gy2 = NufftObj.forward(gx)
            gy2 = NufftObj.forward(x)
            gy2.get()
#         gy2 = NufftObj.k2y(gk)
#             gx2 = NufftObj.adjoint(gy2)
#             gk2 = NufftObj.y2k(gy2)
#         del gy2
#     c = gx2.get()
#         gy=NufftObj.forward(gx)        
        
#     NufftObj.thr.synchronize()
#     gy2.get()
    t_gpu_forward_transfer = (time.time() - t0)/100

    t0= time.time()
    for pp in range(0,100):
#         pass
            gy2 = NufftObj.forward(gx)
#             gy2 = NufftObj.forward(x)
#             gy2.get()
#         gy2 = NufftObj.k2y(gk)
#             gx2 = NufftObj.adjoint(gy2)
#             gk2 = NufftObj.y2k(gy2)
#         del gy2
#     c = gx2.get()
#         gy=NufftObj.forward(gx)        
        
    NufftObj.thr.synchronize() # force synchronize
#     gy2.get()
    t_gpu_forward_no_transfer = (time.time() - t0)/100    
    
    print('gy close? = ', numpy.allclose(y, gy2.get(),  atol=numpy.linalg.norm(y)*1e-6))
#     print("acceleration forward=", t_cpu/t_cl)
    
    return t_cpu_adj, t_cpu_forward, t_gpu_adj_transfer, t_gpu_forward_transfer, t_gpu_adj_no_transfer, t_gpu_forward_no_transfer

        
if __name__ == '__main__':
    t_cpu_adj = []
    t_cpu_forward = []
    t_gpu_adj_transfer = []
    t_gpu_forward_transfer = []
    t_gpu_adj_no_transfer = [] 
    t_gpu_forward_no_transfer = []
    
    tested_batch_numbers = (1,2, 3, 4, 6, 8, 16, 32)
    
    for batch in tested_batch_numbers:
        t1, t2, t3, t4, t5, t6 = test_init(batch)
        t_cpu_adj += [t1, ]
        t_cpu_forward += [t2, ]
        t_gpu_adj_transfer += [t3, ]
        t_gpu_forward_transfer += [t4, ]
        t_gpu_adj_no_transfer += [t5, ] 
        t_gpu_forward_no_transfer += [t6, ]
        
#     The run-times of gpuNUFFT, using the same k-space (Quadro P6000)
#     You must benchmark and record runtimes in MATLAB
    gpuNUFFT_adj = [0.0055539, 0.0069111, 0.0075481, 0.0083244, 0.0098796, 0.011393, 0.020787, 0.036135]
    gpuNUFFT_forward = [0.0055263, 0.0069169, 0.0080937, 0.009501, 0.011558, 0.013507, 0.021263, 0.035416]

    import matplotlib.pyplot
    matplotlib.pyplot.plot(tested_batch_numbers, t_gpu_forward_transfer,'x--k', label='GPU forward (transfer)')
    matplotlib.pyplot.plot(tested_batch_numbers, t_gpu_adj_transfer,'o--k', label='GPU adjoint (transfer)')
    matplotlib.pyplot.plot(tested_batch_numbers, t_gpu_forward_no_transfer, 'x-r', label='GPU forward (no transfer)')
    matplotlib.pyplot.plot(tested_batch_numbers, t_gpu_adj_no_transfer, 'o-r', label='GPU adjoint (no transfer)')
    matplotlib.pyplot.plot(tested_batch_numbers, gpuNUFFT_forward, 'x:b', label='gpuNUFFT forward (transfer)')
    matplotlib.pyplot.plot(tested_batch_numbers, gpuNUFFT_adj, 'o:b', label='gpuNUFFT adjoint (transfer)')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.ylabel('Time (s)')
    matplotlib.pyplot.xlabel('Number of coils')
    matplotlib.pyplot.show()
    
