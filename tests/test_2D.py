    
import os
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import scipy

def test_2D():
    import pkg_resources
    
    DATA_PATH = pkg_resources.resource_filename('pynufft', 'src/data/')
#     PHANTOM_FILE = pkg_resources.resource_filename('pynufft', 'data/phantom_256_256.txt')
    import numpy
    import matplotlib.pyplot
    from ..pynufft import NUFFT_cpu
    # load example image
#     image = numpy.loadtxt(DATA_PATH +'phantom_256_256.txt')
    image = scipy.misc.ascent()
    
    image = scipy.misc.imresize(image, (256,256))
    
    image=image.astype(numpy.float)/numpy.max(image[...])
    #numpy.save('phantom_256_256',image)
    matplotlib.pyplot.imshow(image, cmap=matplotlib.cm.gray)
    matplotlib.pyplot.show()
    print('loading image...')

    
    
    Nd = (256, 256)  # image size
    print('setting image dimension Nd...', Nd)
    Kd = (512, 512)  # k-space size
    print('setting spectrum dimension Kd...', Kd)
    Jd = (6, 6)  # interpolation size
    print('setting interpolation size Jd...', Jd)
    # load k-space points
    # om = numpy.loadtxt(DATA_PATH+'om.txt')
    om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']
    print('setting non-uniform coordinates...')
    matplotlib.pyplot.plot(om[::10,0],om[::10,1],'o')
    matplotlib.pyplot.title('non-uniform coordinates')
    matplotlib.pyplot.xlabel('axis 0')
    matplotlib.pyplot.ylabel('axis 1')
    matplotlib.pyplot.show()

    NufftObj = NUFFT_cpu()
    NufftObj.plan(om, Nd, Kd, Jd)
    
    y = NufftObj.forward(image)
    print('setting non-uniform data')
    print('y is an (M,) list',type(y), y.shape)
    
#     kspectrum = NufftObj.xx2k( NufftObj.solve(y,solver='bicgstab',maxiter = 100))
    image_restore = NufftObj.solve(y, solver='cg',maxiter=10)
    shifted_kspectrum = numpy.fft.fftshift( numpy.fft.fftn( numpy.fft.fftshift(image_restore)))
    print('getting the k-space spectrum, shape =',shifted_kspectrum.shape)
    print('Showing the shifted k-space spectrum')
    
    matplotlib.pyplot.imshow( shifted_kspectrum.real, cmap = matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=-100, vmax=100))
    matplotlib.pyplot.title('shifted k-space spectrum')
    matplotlib.pyplot.show()
    image2 = NufftObj.solve(y, 'dc', maxiter = 25)
    image3 = NufftObj.solve(y, 'L1TVLAD',maxiter=100, rho= 1)
    image4 = NufftObj.solve(   y,'L1TVOLS',maxiter=100, rho= 1)
    matplotlib.pyplot.subplot(1,3,1)
    matplotlib.pyplot.imshow(image, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
    matplotlib.pyplot.subplot(1,3,2)
    matplotlib.pyplot.imshow(image3.real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
    matplotlib.pyplot.subplot(1,3,3)
    matplotlib.pyplot.imshow(image4.real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
    matplotlib.pyplot.show()  
  
#     matplotlib.pyplot.imshow(image2.real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
#     matplotlib.pyplot.show()
    maxiter =25
    counter = 1
    for solver in ('dc','bicg','bicgstab','cg', 'gmres','lgmres',  'lsmr', 'lsqr'):
        if 'lsqr' == solver:
            image2 = NufftObj.solve(y, solver,iter_lim=maxiter)
        else:
            image2 = NufftObj.solve(y, solver,maxiter=maxiter)
#     image2 = NufftObj.solve(y, solver='bicgstab',maxiter=30)
        matplotlib.pyplot.subplot(2,4,counter)
        matplotlib.pyplot.imshow(image2.real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
        matplotlib.pyplot.title(solver)
        print(counter, solver)
        counter += 1
    matplotlib.pyplot.show()

# def test_asoperator():
#     
#     import pkg_resources
#     
#     DATA_PATH = pkg_resources.resource_filename('pynufft', 'data/')
# #     PHANTOM_FILE = pkg_resources.resource_filename('pynufft', 'data/phantom_256_256.txt')
#     import numpy
#     import matplotlib.pyplot
#     # load example image
# #     image = numpy.loadtxt(DATA_PATH +'phantom_256_256.txt')
#     image = scipy.misc.ascent()
#     
#     image = scipy.misc.imresize(image, (256,256))
#     
#     image=image.astype(numpy.float)/numpy.max(image[...])
#     #numpy.save('phantom_256_256',image)
#     matplotlib.pyplot.imshow(image, cmap=matplotlib.cm.gray)
#     matplotlib.pyplot.show()
#     print('loading image...')
# 
#     
#     
#     Nd = (256, 256)  # image size
#     print('setting image dimension Nd...', Nd)
#     Kd = (512, 512)  # k-space size
#     print('setting spectrum dimension Kd...', Kd)
#     Jd = (6, 6)  # interpolation size
#     print('setting interpolation size Jd...', Jd)
#     # load k-space points
#     # om = numpy.loadtxt(DATA_PATH+'om.txt')
#     om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']
#     print('setting non-uniform coordinates...')
#     matplotlib.pyplot.plot(om[::10,0],om[::10,1],'o')
#     matplotlib.pyplot.title('non-uniform coordinates')
#     matplotlib.pyplot.xlabel('axis 0')
#     matplotlib.pyplot.ylabel('axis 1')
#     matplotlib.pyplot.show()
# 
#     NufftObj = NUFFT()
#     NufftObj.plan(om, Nd, Kd, Jd)
#         
#     print('NufftObj.dtype=',NufftObj.dtype, '  NufftObj.shape=',NufftObj.shape)
#     
#     y1 = NufftObj.forward(image)
#     x_vec= numpy.reshape(image,  (numpy.prod(NufftObj.st['Nd']), ) , order='F')
#     
#     y2 = NufftObj._matvec(x_vec)
#     
#     A = scipy.sparse.linalg.LinearOperator(NufftObj.shape, matvec=NufftObj._matvec, rmatvec=NufftObj._adjoint, dtype=NufftObj.dtype)
# #     scipy.sparse.linalg.aslinearoperator(A)
#     
#     print(type(A))
# #     y1 = A.matvec(x_vec)
#     print('y1.shape',numpy.shape(y1))  
#     import time
#     t0=time.time()
#     KKK = scipy.sparse.linalg.lsqr(A, y1, )
#     print(numpy.shape(KKK))  
#     
#     
#     print(time.time() - t0)
#     
#     x2 = numpy.reshape(KKK[0], NufftObj.st['Nd'], order='F')
#     
#     
#     matplotlib.pyplot.subplot(2,1,1)
#     matplotlib.pyplot.imshow(x2.real,cmap=matplotlib.cm.gray)
#     matplotlib.pyplot.subplot(2,1,2)
#     matplotlib.pyplot.imshow(image,cmap=matplotlib.cm.gray)
#     
#     matplotlib.pyplot.show()
#     
#     
#     print('y1 y2 close? ', numpy.allclose(y1, y2))
        

if __name__ == '__main__':
    """
    Test the module pynufft
    """
    test_2D()
#     test_asoperator()
#     test_installation()