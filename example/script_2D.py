import numpy 
import scipy.misc
import matplotlib.pyplot 

from pynufft.pynufft import NUFFT_cpu







# load k-space points
import pkg_resources
DATA_PATH = pkg_resources.resource_filename('pynufft', './src/data/')
om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']
print(om)
print('setting non-uniform coordinates...')
matplotlib.pyplot.plot(om[::10,0],om[::10,1],'o')
matplotlib.pyplot.title('non-uniform coordinates')
matplotlib.pyplot.xlabel('axis 0')
matplotlib.pyplot.ylabel('axis 1')
matplotlib.pyplot.show()

NufftObj = NUFFT_cpu()

Nd = (256, 256)  # image size
print('setting image dimension Nd...', Nd)
Kd = (512, 512)  # k-space size
print('setting spectrum dimension Kd...', Kd)
Jd = (6, 6)  # interpolation size
print('setting interpolation size Jd...', Jd)

NufftObj.plan(om, Nd, Kd, Jd)

image = scipy.misc.ascent()
image = scipy.misc.imresize(image, (256,256))
image=image/numpy.max(image[...])

print('loading image...')

matplotlib.pyplot.imshow(image.real, cmap=matplotlib.cm.gray)
matplotlib.pyplot.show()


y = NufftObj.forward(image)
print('setting non-uniform data')
print('y is an (M,) list',type(y), y.shape)


matplotlib.pyplot.subplot(2,2,1)
image0 = NufftObj.solve(y, solver='cg',maxiter=50)
matplotlib.pyplot.title('Restored image (cg)')
matplotlib.pyplot.imshow(image0.real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))


matplotlib.pyplot.subplot(2,2,2)
image2 = NufftObj.adjoint(y )
matplotlib.pyplot.imshow(image2.real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=5))
matplotlib.pyplot.title('Adjoint transform')


matplotlib.pyplot.subplot(2,2,3)
image3 = NufftObj.solve(y, solver='L1TVOLS',maxiter=50,rho=0.1)
matplotlib.pyplot.title('L1TV OLS')
matplotlib.pyplot.imshow(image3.real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))

matplotlib.pyplot.subplot(2,2,4)
image4 = NufftObj.solve(y, solver='L1TVLAD',maxiter=50,rho=0.1)
matplotlib.pyplot.title('L1TV LAD')
matplotlib.pyplot.imshow(image4.real, cmap=matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1))
matplotlib.pyplot.show()


shifted_kspectrum = numpy.fft.fftshift(numpy.fft.fftn(numpy.fft.fftshift(image0)))
#     print('getting the k-space spectrum, shape =',shifted_kspectrum.shape)
print('Showing the shifted k-space spectrum')

matplotlib.pyplot.imshow( shifted_kspectrum.real, cmap = matplotlib.cm.gray, norm=matplotlib.colors.Normalize(vmin=-100, vmax=100))
matplotlib.pyplot.title('shifted k-space spectrum')
matplotlib.pyplot.show()