import numpy 
import scipy.misc
import matplotlib.pyplot 

from pynufft import NUFFT_cpu







# load k-space points
import pkg_resources
DATA_PATH = pkg_resources.resource_filename('pynufft', './src/data/')
om = numpy.load(DATA_PATH+'om2D.npz')['arr_0']

# om = numpy.random.randn(120000, 2)
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
image=image*1.0/numpy.max(image[...])

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


W0 = numpy.ones((NufftObj.st['M'], ))


# W_x = NufftObj.xx2k( NufftObj.adjoint(NufftObj.forward(NufftObj.k2xx(W0))))
# W_y =  NufftObj.xx2k(NufftObj.x2xx(NufftObj.adjoint(NufftObj.k2y(W0))))
W =  NufftObj.xx2k(NufftObj.adjoint(W0))

# W =   NufftObj.y2k(W0)
# matplotlib.pyplot.subplot(1,)
matplotlib.pyplot.imshow(numpy.real((W*W.conj())**0.5))
matplotlib.pyplot.title('Ueckers inverse function (real)')
# matplotlib.pyplot.subplot(1,2,2)
# matplotlib.pyplot.imshow(W.imag)
# matplotlib.pyplot.title('Ueckers inverse function (imaginary)')
matplotlib.pyplot.show()


p0 = NufftObj.adjoint(NufftObj.forward(image))
p1 = NufftObj.k2xx((W.conj()*W)**0.5*NufftObj.xx2k(image))

print('error between Toeplitz and Inverse reconstruction', numpy.linalg.norm( p1 - p0)/ numpy.linalg.norm(p0))


matplotlib.pyplot.subplot(1,3,1)
matplotlib.pyplot.imshow(numpy.real(p0 ))
matplotlib.pyplot.title('Toeplitz')
matplotlib.pyplot.subplot(1,3,2)
matplotlib.pyplot.imshow(numpy.real(p1))
matplotlib.pyplot.title('Ueckers inverse function')
matplotlib.pyplot.subplot(1,3,3)
matplotlib.pyplot.imshow(numpy.abs(p0 - p1)/ numpy.abs(p1))
matplotlib.pyplot.title('Difference')
matplotlib.pyplot.show()

