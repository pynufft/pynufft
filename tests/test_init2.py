from .. import NUFFT_coil, NUFFT_cpu

import numpy
dtype = numpy.complex64

def test_init2():
    import cProfile
    import numpy
#     import matplotlib.pyplot
    import copy
#     cm = matplotlib.cm.gray
    # load example image
    import pkg_resources
    
    DATA_PATH = pkg_resources.resource_filename('pynufft', 'src/data/')
#     PHANTOM_FILE = pkg_resources.resource_filename('pynufft', 'data/phantom_256_256.txt')
    import numpy
    
#     import matplotlib.pyplot
    
    import scipy

    image = scipy.misc.ascent()    
    image = scipy.misc.imresize(image, (256,256))
    
    image=image.astype(numpy.float)/numpy.max(image[...])

    Nd = (256, 256)  # image space size
    Kd = (512, 512)  # k-space size
    Jd = (6,6)  # interpolation size

    # load k-space points
    om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']
    M = om.shape[0]
    nfft0 = NUFFT_cpu()  # CPU
    
    nfft0.plan(om, Nd, Kd, Jd)

    nfft = NUFFT_coil()  # CPU
    Nc = 3
    from . import multicoil_solver
    fake_coil =  multicoil_solver.create_fake_coils(Nd[0], Nc)
    
    image_stack = numpy.ones(Nd + (Nc,), dtype = numpy.complex)
    for pp in range(0, Nc):
        image_stack[...,pp] = fake_coil[pp]
    
    nfft.plan1(om, Nd, Kd, Jd, ft_axes =  (0,1), image_stack= image_stack)

    print('error between current and old interpolators=', 
          scipy.sparse.linalg.norm(nfft.sp[0:M,:] - nfft0.sp)/scipy.sparse.linalg.norm(nfft0.sp))
    
#     NufftObj.offload(API = 'ocl',   platform_number = 0, device_number = 0)
    y = nfft.forward(image)
    
    

#     try:
    for coil in range(0, Nc): 
        x2 =  nfft0.solve(y[M*coil: M*(coil + 1)], 'lsmr', maxiter= 20)
        import matplotlib.pyplot
        matplotlib.pyplot.subplot(3, 3, 1+coil*3)
        matplotlib.pyplot.imshow( x2.imag, cmap= matplotlib.cm.gray)#,    vmin=-1.,vmax=1.)
        matplotlib.pyplot.subplot(3, 3,2+coil*3)
        matplotlib.pyplot.imshow(image.imag, cmap= matplotlib.cm.gray)#,vmin=-1.,vmax=1.)
        matplotlib.pyplot.subplot(3, 3,3+coil*3)
        matplotlib.pyplot.imshow((image_stack[...,coil]*image).imag, cmap= matplotlib.cm.gray,)#vmin=-1.,vmax=1.)
    matplotlib.pyplot.show()
    for coil in range(0, Nc): 
        x2 =  nfft0.solve(y[M*coil: M*(coil + 1)], 'lsmr', maxiter= 20)
        import matplotlib.pyplot
        matplotlib.pyplot.subplot(3, 3, 1+coil*3)
        matplotlib.pyplot.imshow( x2.real, cmap= matplotlib.cm.gray)#, vmin=-1.,vmax=1.)
        matplotlib.pyplot.subplot(3, 3,2+coil*3)
        matplotlib.pyplot.imshow(image.real, cmap= matplotlib.cm.gray)#, vmin=-1.,vmax=1.)
        matplotlib.pyplot.subplot(3, 3,3+coil*3)
        matplotlib.pyplot.imshow((image_stack[...,coil]*image).real, cmap= matplotlib.cm.gray)#, vmin=-1.,vmax=1.)
    matplotlib.pyplot.show()        
 
#     except:
#         print("no graphics")
# if __name__ == '__main__':
#     test_init()    