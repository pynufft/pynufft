'''@package docstring
@author: Jyh-Miin Lin
@address: jyhmiinlin@gmail.com 
@date: 2016/1/18
   
    Pythonic non-uniform fast Fourier transform (NUFFT) with FFTW
    Please cite http://scitation.aip.org/content/aapm/journal/medphys/42/10/10.1118/1.4929560 
    
    This algorithm was proposed by 
    Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max interpolation. IEEE Trans Signal Process 2003;51(2):560-574.

    Installation: 
    pip install pynufft
    
    test:
    import pynufft.pynufft_pyfftw
    pynufft.pynufft_pyfftw.test_1D()
    pynufft.pynufft_pyfftw.test_2D()
     
    Check test_1D() and test_2D() for usage
       

'''

import numpy
import scipy.sparse
import numpy.fft
import pynufft

dtype = numpy.complex64

import scipy.signal

import scipy.linalg

try: 
    xrange 
except NameError: 
    xrange = range

# try:
#     import pyfftw
# except:
#     print('fftw and pyfftw are needed')


class pynufft_pyfftw(pynufft.pynufft):
  
    def __init__(self,om, Nd, Kd,Jd, n_shift):
        pynufft.pynufft.__init__(self,om, Nd, Kd,Jd,n_shift )
        self.initialize_pyfftw() # the initialize pyfftw
    def initialize_pyfftw(self):
        self.pyfftw_flag = 0
        try:
            import pyfftw # import python wrapper for FFTW
                            
            pyfftw.interfaces.cache.enable()
            pyfftw.interfaces.cache.set_keepalive_time(60) # keep live 60 seconds
            self.pyfftw_flag = 1
            self.threads=2
#             if self.debug > 0:
            if self.debug == 1:
                print('use pyfftw')
        except:
#             if self.debug > 0:
            print('no pyfftw, use scipy/numpy fft')
            self.pyfftw_flag = 0
    def emb_fftn(self, input_x, output_dim, act_axes):
        '''
        The abstraction for fast Fourier transform
        '''
        output_x=numpy.zeros(output_dim, dtype=dtype)

        output_x[pynufft.crop_slice_ind(input_x.shape)] = input_x

#         if self.pyfftw_flag ==1:
# 
        output_x=pyfftw.interfaces.scipy_fftpack.fftn(output_x, output_dim, act_axes, 
                                                          threads=self.threads,overwrite_x=True)
# 
        if self.debug == 1:
            print('  fftw')
        return output_x
  
    def emb_ifftn(self, input_x, output_dim, act_axes):
        '''
        The abstraction for inverse fast Fourier transform
        '''
        output_x=numpy.zeros(output_dim, dtype=dtype)

        output_x[pynufft.crop_slice_ind(input_x.shape)] = input_x 

#         if self.pyfftw_flag == 1:
        output_x=pyfftw.interfaces.scipy_fftpack.ifftn(output_x, output_dim, act_axes, 
                                                          threads=self.threads,overwrite_x=True)
        if self.debug == 1:
            print('  fftw')

 
        return output_x    
    
def test_1D():
# import several modules
    import numpy 
    import matplotlib.pyplot

    import os
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)

#create 1D curve from 2D image
    image = numpy.loadtxt(dir_path+'/phantom_256_256.txt') 
    profile_1d = image[:,128]
#determine the location of samples
    om = numpy.loadtxt(dir_path+'/om1D.txt')[0:]
    om = numpy.reshape(om,(numpy.size(om),1),order='F')
# reconstruction parameters
    Nd =(256,) # image space size
    Kd =(512,) # k-space size
      
    Jd =(6,) # interpolation size
# initiation of the object
    NufftObj = pynufft_pyfftw(om, Nd,Kd,Jd,None)
# simulate "data"
    data= NufftObj.forward(profile_1d)
#adjoint(reverse) of the forward transform


    NufftObj.pipe_density() # recalculate the density compensation
    
    profile_distorted= NufftObj.adjoint(data*NufftObj.st['W'])[:,0]

 
#Showing histogram of sampling locations
#     matplotlib.pyplot.hist(om,20)
#     matplotlib.pyplot.title('histogram of the sampling locations')
#     matplotlib.pyplot.show()
#show reconstruction
    matplotlib.pyplot.subplot(1,2,1)
 
    matplotlib.pyplot.plot(profile_1d)
    matplotlib.pyplot.title('original') 
    matplotlib.pyplot.ylim([0,1]) 
            

             
    matplotlib.pyplot.subplot(1,2,2)
    matplotlib.pyplot.plot(numpy.abs(profile_distorted)) 
    matplotlib.pyplot.title('profile_distorted')
 
    matplotlib.pyplot.show()  
         
    
def test_2D():
 
    import numpy 
    import matplotlib.pyplot
#     import os
#     path = os.path.dirname(self.__file__)
    import os
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
#     print(dir_path)
    cm = matplotlib.cm.gray
    # load example image    
 
    image = numpy.loadtxt(dir_path+'/phantom_256_256.txt') 
    image[128,128]= 1.0   
    Nd =(256,256) # image space size
    Kd =(512,512) # k-space size   
    Jd =(6,6) # interpolator size
     
    # load k-space points
    om = numpy.loadtxt(dir_path+'/om.txt')
     
    #create object
  
#      
    NufftObj = pynufft_pyfftw(om, Nd,Kd,Jd, None)   
     
#     NufftObj = pynufft.pynufft.pynufft(om, Nd,Kd,Jd, None)   

    data= NufftObj.forward(image )
    
    NufftObj.pipe_density() # recalculate the density compensation
    
    image_blur = NufftObj.adjoint(data*NufftObj.st['W'])
#     image_recon = Normalize(image_recon)
 
    matplotlib.pyplot.plot(om[:,0],om[:,1],'x')
    matplotlib.pyplot.show()
 
    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0) 
    norm2=matplotlib.colors.Normalize(vmin=0.0, vmax=5.0e-2)
    # display images
    matplotlib.pyplot.subplot(2,2,1)
    matplotlib.pyplot.imshow(image,
                             norm = norm,cmap =cm,interpolation = 'nearest')
    matplotlib.pyplot.title('true image')   
   
    matplotlib.pyplot.subplot(2,2,2)
    matplotlib.pyplot.imshow(image_blur[:,:,0].real,
                              norm = norm2,cmap= cm,interpolation = 'nearest')
    matplotlib.pyplot.title('blurred image') 
 
     
    matplotlib.pyplot.show()      

if __name__ == '__main__':
#     import cProfile
#     test_1D()
    test_2D()   
          