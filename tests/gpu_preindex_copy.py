
def cpu_preindex_copy(image, Nd, Kd): 

    from pynufft.src._helper import helper
    
    
    copy_in, copy_out, ele = helper.preindex_copy(Nd, Kd)
    print(copy_in, copy_out)
    
    

    image2 = numpy.zeros(Kd)
    import time
    t0 = time.time()
    for pp in range(0, 100):
        image2.ravel()[copy_out] = image.ravel()[copy_in]
    t1 = time.time()
    print('cpu_time = ', (t1 - t0)/100)
    return image2
    
def gpu_preindex_copy(image, Nd, Kd):
    import reikna.cluda as cluda
    from pynufft.src.re_subroutine import create_kernel_sets
    kernel_sets = create_kernel_sets('ocl')
               
    from pynufft.src._helper import helper
    
    
    copy_in, copy_out, ele = helper.preindex_copy(Nd, Kd)    
    
    api = cluda.ocl_api()
    device = api.get_platforms()[1].get_devices()[0]
    thr = api.Thread(device)
    prg = thr.compile(kernel_sets, 
                                render_kwds=dict(LL =  str(2)), 
                                fast_math=False)
    g_image = thr.to_device(image.astype(numpy.complex64))
    g_image3 = thr.array(Kd, dtype = numpy.complex64).fill(0.0)
    prodNd = numpy.prod(Nd)
    prodKd = numpy.prod(Kd)
    dim = numpy.uint32(len(Nd))
    print(dim, prodNd, prodKd)
    res_Nd = ()
    res_Kd = ()
    accum_prodnd = 1
    accum_prodkd = 1
    for pp in range(0, dim):
        """
        Nd:    2    3    4    5
        prodNd 120
        dim: 0    1    2    3
            60    20    5    1
        
        """
        accum_prodnd *= Nd[pp]
        accum_prodkd *= Kd[pp]
        res_Nd += (prodNd/ accum_prodnd,)
        res_Kd += (prodKd/ accum_prodkd,)
          
        
    print(res_Nd, res_Kd)
    g_Nd = thr.to_device(numpy.array(res_Nd, dtype = numpy.uint32))
    g_Kd = thr.to_device(numpy.array(res_Kd, dtype = numpy.uint32))
    g_invNd = thr.to_device(1/numpy.array(res_Nd, dtype = numpy.float32))
    
    g_in =  thr.to_device(copy_in.astype(numpy.uint32))
    g_out =  thr.to_device(copy_out.astype(numpy.uint32))
    
    print(g_Nd.get())
    print(g_Kd.get())
    import time
    t0 = time.time()
    for pp in range(0, 100):
        prg.cTensorCopy(dim, g_Nd, g_Kd, g_invNd, g_image, g_image3, local_size = None, global_size = int(prodNd))
    
    t1 = time.time()
    print('gpu_time = ', (t1 - t0)/100)
    thr.synchronize()
    return g_image3
    
    
    
Nd = (512,512)
Kd = (1024,1024)

import scipy.misc, numpy
image = scipy.misc.ascent()
image = scipy.misc.imresize(image, Nd)
image = image/numpy.max(abs(image.ravel()))
# image = numpy.abs(numpy.random.randn(128,128,128)).astype(numpy.complex64)

image2 = cpu_preindex_copy(image, Nd, Kd)
image3 = gpu_preindex_copy(image, Nd, Kd).get()


import matplotlib.pyplot

matplotlib.pyplot.subplot(1,3,1)
matplotlib.pyplot.imshow(image[:,:].real)
 
matplotlib.pyplot.subplot(1,3,2)
matplotlib.pyplot.imshow(image2[:,:].real)
 
 
matplotlib.pyplot.subplot(1,3,3)
matplotlib.pyplot.imshow(image3[:,:].real)
# 
# matplotlib.pyplot.subplot(1,3,1)
# matplotlib.pyplot.imshow(image[64,:,:].real)
#  
# matplotlib.pyplot.subplot(1,3,2)
# matplotlib.pyplot.imshow(image2[64,:,:].real)
#  
#  
# matplotlib.pyplot.subplot(1,3,3)
# matplotlib.pyplot.imshow(image3[64,:,:].real)


matplotlib.pyplot.show()


