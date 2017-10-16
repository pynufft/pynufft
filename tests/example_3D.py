import numpy 
import matplotlib.pyplot as pyplot
from matplotlib import cm
gray = cm.gray
    

def example_3D():

 
    import pkg_resources
    
    DATA_PATH = pkg_resources.resource_filename('pynufft', './src/data/')   
    
    image = numpy.load(DATA_PATH +'phantom_3D_128_128_128.npz')['arr_0'][0::2, 0::2, 0::2]


    pyplot.imshow(numpy.abs(image[:,:,32]), label='original signal',cmap=gray)
    pyplot.show()

     
    Nd = (64,64,64) # time grid, tuple
    Kd = (64,64,64) # frequency grid, tuple
    Jd = (1,1,1) # interpolator 
#     om=       numpy.load(DATA_PATH+'om3D.npz')['arr_0']
    om = numpy.random.randn(15120,3)
    print(om.shape)
    from ..pynufft import NUFFT_cpu, NUFFT_hsa
    NufftObj = NUFFT_cpu()
    
    
    NufftObj.plan(om, Nd, Kd, Jd)

    kspace =NufftObj.forward(image)
    
    restore_image = NufftObj.solve(kspace,'cg', maxiter=200)
    
    restore_image1 = NufftObj.solve(kspace,'L1TVLAD', maxiter=200,rho=0.1)
# 
    restore_image2 = NufftObj.solve(kspace,'L1TVOLS', maxiter=200,rho=0.1)
    pyplot.subplot(2,2,1)
    pyplot.imshow(numpy.abs(image[:,:,32]), label='original signal',cmap=gray)
    pyplot.title('original')    
    pyplot.subplot(2,2,2)
    pyplot.imshow(numpy.abs(restore_image1[:,:,32]), label='L1TVLAD',cmap=gray)
    pyplot.title('L1TVLAD')
    
    pyplot.subplot(2,2,3)
    pyplot.imshow(numpy.abs(restore_image2[:,:,32]), label='L1TVOLS',cmap=gray)
    pyplot.title('L1TVOLS')
        

    
    pyplot.subplot(2,2,4)
    pyplot.imshow(numpy.abs(restore_image[:,:,32]), label='CG',cmap=gray)
    pyplot.title('CG')
#     pyplot.legend([im1, im im4])
    
    
    pyplot.show()


