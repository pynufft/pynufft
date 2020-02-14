import os
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import scipy

def test_2D_batch():
    import pkg_resources
    
    DATA_PATH = pkg_resources.resource_filename('pynufft', 'src/data/')
#     PHANTOM_FILE = pkg_resources.resource_filename('pynufft', 'data/phantom_256_256.txt')
    import numpy
    import matplotlib.pyplot
    from pynufft import NUFFT_cpu
    # load example image
#     image = numpy.loadtxt(DATA_PATH +'phantom_256_256.txt')
    image = scipy.misc.ascent()[::2,::2]
    x=image.astype(numpy.float)/numpy.max(image[...])
    #numpy.save('phantom_256_256',image)
    x = numpy.reshape(x, (256,256,1))
    batch = 3
    image = numpy.broadcast_to(x, (256,256,batch)).copy()
    matplotlib.pyplot.imshow(image[..., 0], cmap=matplotlib.cm.gray)
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
    NufftObj.plan(om, Nd, Kd, Jd, batch = batch)
    
    y = NufftObj.forward(image)
    print('setting non-uniform data')
    print('y is an (M,) list',type(y), y.shape)
    
#     kspectrum = NufftObj.xx2k( NufftObj.solve(y,solver='bicgstab',maxiter = 100))
    image_restore = NufftObj.solve(y, solver='cg',maxiter=10)
    shifted_kspectrum = numpy.fft.fftshift( numpy.fft.fftn( numpy.fft.fftshift(image_restore, axes = (0,1)), axes = (0,1)), axes = (0,1))
    print('getting the k-space spectrum, shape =',shifted_kspectrum.shape)
    print('Showing the shifted k-space spectrum')
    
    matplotlib.pyplot.imshow( shifted_kspectrum[...,0].real, cmap = matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=-100, vmax=100))
    matplotlib.pyplot.title('shifted k-space spectrum')
    matplotlib.pyplot.show()
#     image4 = NufftObj.solve(y,'L1TVOLS',maxiter=100, rho= 1)
#     image2 = NufftObj.solve(y, 'dc', maxiter = 25)
#     image3 = NufftObj.solve(y, 'cg', maxiter = 25)
#     
#     matplotlib.pyplot.subplot(1,3,1)
#     matplotlib.pyplot.imshow(image2[...,1].real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
#     matplotlib.pyplot.title('dc')
#     matplotlib.pyplot.subplot(1,3,2)
#     matplotlib.pyplot.imshow(image3[...,1].real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
#     matplotlib.pyplot.title('cg')
#     matplotlib.pyplot.subplot(1,3, 3)
#     matplotlib.pyplot.imshow(image4.real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
#     matplotlib.pyplot.title('L1TVOLS')
#     matplotlib.pyplot.show()  
  
#     matplotlib.pyplot.imshow(image2.real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
#     matplotlib.pyplot.show()
    maxiter =25
    counter = 1
    for solver in ('lsqr', 'dc','bicg','bicgstab','cg', 'gmres','lgmres'):
        print(counter, solver)
        if 'lsqr' == solver:
            image2 = NufftObj.solve(y, solver,iter_lim=maxiter)
        else:
            image2 = NufftObj.solve(y, solver,maxiter=maxiter)
#     image2 = NufftObj.solve(y, solver='bicgstab',maxiter=30)
        matplotlib.pyplot.subplot(2,4,counter)
        matplotlib.pyplot.imshow(image2[...,1].real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
        matplotlib.pyplot.title(solver)
#         print(counter, solver)
        counter += 1
    matplotlib.pyplot.show()


if __name__ == '__main__':
    """
    Test the module pynufft
    """
    test_2D_batch()
#     test_asoperator()
#     test_installation()
