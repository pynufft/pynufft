
    
def gpu_preindex_multiply(image, vec, axis):
    import reikna.cluda as cluda
    from pynufft.src.re_subroutine import create_kernel_sets
    kernel_sets = create_kernel_sets('ocl')
               
    from pynufft.src._helper import helper
    
    
    Nd = image.shape
    Nd_elements = tuple( numpy.prod(Nd[dd+1:]) for dd in range(len(Nd)))  
    invNd_elements = 1.0/numpy.asarray(Nd_elements)
    print(Nd_elements,invNd_elements)
    api = cluda.ocl_api()
    device = api.get_platforms()[0].get_devices()[0]
    thr = api.Thread(device)
    prg = thr.compile(kernel_sets, 
                                render_kwds=dict(LL =  str(2)), 
                                fast_math=False)
    g_image = thr.to_device(image.astype(numpy.complex64))
#     g_image3 = thr.array(Nd, dtype = numpy.complex64).fill(0.0)
    g_vec = thr.to_device(vec.astype(numpy.float32))
    prodNd = numpy.prod(Nd)
#     import time
#     t0 = time.time()
#     for pp in range(0, 100):

    g_Nd = thr.to_device(numpy.asarray(Nd, dtype = numpy.uint32))
    g_Nd_elements = thr.to_device(numpy.asarray(Nd_elements, dtype = numpy.uint32))
    g_invNd_elements = thr.to_device(numpy.asarray(invNd_elements, dtype = numpy.float32))
    vec2 = numpy.zeros((numpy.sum(Nd), ), dtype = numpy.float32)
    vec2[0:Nd[0]] = vec
    vec2[Nd[0]:(Nd[0] + Nd[1])] = vec
    g_vec = thr.to_device(vec2)
    prg.cTensorMultiply(numpy.uint32(1), 
                                    numpy.uint32(len(Nd)),
                                    g_Nd,
                                    g_Nd_elements,
                                    g_invNd_elements,
                                    g_vec, 
                                    g_image, 
                                    local_size = None, global_size = int(prodNd))
    
#     t1 = time.time()
#     print('gpu_time = ', (t1 - t0)/100)
    thr.synchronize()
    return g_image
    
    
    
Nd = (512,512)
Kd = (1024,1024)

import scipy.misc, numpy
image = scipy.misc.ascent()
image = scipy.misc.imresize(image, Nd)*(1.0+1.0j)
image = image/numpy.max(abs(image.ravel()))
# image = numpy.abs(numpy.random.randn(128,128,128)).astype(numpy.complex64)
axis = 1
# vec = numpy.ones((Nd[axis], ))
# vec[int(Nd[axis]/2):] = 0.0
vec = numpy.cos( (numpy.arange(Nd[axis]) /Nd[axis] - 0.5 )*3 )
image3 = gpu_preindex_multiply(image, vec, axis).get()


import matplotlib.pyplot

matplotlib.pyplot.subplot(1,3,1)
matplotlib.pyplot.imshow(image[:,:].real)
 
matplotlib.pyplot.subplot(1,3,2)
matplotlib.pyplot.imshow(image3[:,:].real)
 
 
matplotlib.pyplot.subplot(1,3,3)
matplotlib.pyplot.plot(vec)
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


