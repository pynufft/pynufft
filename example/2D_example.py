import numpy 
import scipy
import matplotlib.pyplot as pyplot

try:
    '''
    pip install pynufft
    '''
    from pynufft.pynufft import pynufft
    import pkg_resources

    DATA_PATH = pkg_resources.resource_filename('pynufft', '../data/')
except:
    print('warning: pynufft  not found in system library')
    print('Try to import the local pynufft now')
    import sys
    sys.path.append('../pynufft')
    from pynufft import pynufft
    import pkg_resources

    DATA_PATH = pkg_resources.resource_filename('pynufft', '../data/')



#     PHANTOM_FILE = pkg_resources.resource_filename('pynufft', '../data/phantom_256_256.txt')
import numpy
import matplotlib.pyplot
# load example image
# image = numpy.load(DATA_PATH +'phantom_256_256.npz')['arr_0']
image = scipy.misc.face(gray=True)
image = scipy.misc.imresize(image, (256,256))
image=image/numpy.max(image[...])
#numpy.save('phantom_256_256',image)

print('loading image...')

matplotlib.pyplot.imshow(image.real, cmap=matplotlib.cm.gray)
matplotlib.pyplot.show()


Nd = (256, 256)  # image size
print('setting image dimension Nd...', Nd)
Kd = (512, 512)  # k-space size
print('setting spectrum dimension Kd...', Kd)
Jd = (6, 6)  # interpolation size
print('setting interpolation size Jd...', Jd)
# load k-space points
om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']
print('setting non-uniform coordinates...')
matplotlib.pyplot.plot(om[::10,0],om[::10,1],'o')
matplotlib.pyplot.title('non-uniform coordinates')
matplotlib.pyplot.xlabel('axis 0')
matplotlib.pyplot.ylabel('axis 1')
matplotlib.pyplot.show()

NufftObj = pynufft()
NufftObj.plan(om, Nd, Kd, Jd)

y = NufftObj.forward(image)
print('setting non-uniform data')
print('y is an (M,) list',type(y), y.shape)

kspectrum = NufftObj.y2k_DC(y)
shifted_kspectrum = numpy.fft.fftshift(kspectrum, axes=(0,1))
print('getting the k-space spectrum, shape =',kspectrum.shape)
print('Showing the shifted k-space spectrum')

matplotlib.pyplot.imshow( shifted_kspectrum.real, cmap = matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=-100, vmax=100))
matplotlib.pyplot.title('shifted k-space spectrum')
matplotlib.pyplot.show()

matplotlib.pyplot.subplot(1,2,1)

image2 = NufftObj.adjoint(y * NufftObj.st['W'])

matplotlib.pyplot.imshow(image2.real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
matplotlib.pyplot.subplot(1,2,2)
image2 = NufftObj.adjoint(y )

matplotlib.pyplot.imshow(image2.real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=5))
matplotlib.pyplot.show()

