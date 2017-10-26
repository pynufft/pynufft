special_license='''
The license of the 3D Shepp-Logan phantom:
Copyright (c) 2006, Matthias Schabel 
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are 
met:

* Redistributions of source code must retain the above copyright 
notice, this list of conditions and the following disclaimer. 
* Redistributions in binary form must reproduce the above copyright 
notice, this list of conditions and the following disclaimer in 
the documentation and/or other materials provided with the distribution 
* Neither the name of the University of Utah Department of Radiology nor the names 
of its contributors may be used to endorse or promote products derived 
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
'''
import numpy 
import matplotlib.pyplot as pyplot
from matplotlib import cm
gray = cm.gray
    
  
import pkg_resources
DATA_PATH = pkg_resources.resource_filename('pynufft', './src/data/')   
image = numpy.load(DATA_PATH +'phantom_3D_128_128_128.npz')['arr_0'][0::2, 0::2, 0::2]
print(special_license)

pyplot.imshow(numpy.abs(image[:,:,32]), label='original signal',cmap=gray)
pyplot.show()

 
Nd = (64,64,64) # time grid, tuple
Kd = (64,64,64) # frequency grid, tuple
Jd = (1,1,1) # interpolator 
#     om=       numpy.load(DATA_PATH+'om3D.npz')['arr_0']
om = numpy.random.randn(151200,3)*2
print(om.shape)
from pynufft.pynufft import NUFFT_cpu, NUFFT_hsa
NufftObj = NUFFT_cpu()


NufftObj.plan(om, Nd, Kd, Jd)

kspace =NufftObj.forward(image)

restore_image = NufftObj.solve(kspace,'cg', maxiter=500)

restore_image1 = NufftObj.solve(kspace,'L1TVLAD', maxiter=500,rho=0.1)
# 
restore_image2 = NufftObj.solve(kspace,'L1TVOLS', maxiter=500,rho=0.1)
pyplot.subplot(2,2,1)
pyplot.imshow(numpy.real(image[:,:,32]), label='original signal',cmap=gray)
pyplot.title('original')    
pyplot.subplot(2,2,2)
pyplot.imshow(numpy.real(restore_image1[:,:,32]), label='L1TVLAD',cmap=gray)
pyplot.title('L1TVLAD')

pyplot.subplot(2,2,3)
pyplot.imshow(numpy.real(restore_image2[:,:,32]), label='L1TVOLS',cmap=gray)
pyplot.title('L1TVOLS')
    


pyplot.subplot(2,2,4)
pyplot.imshow(numpy.real(restore_image[:,:,32]), label='CG',cmap=gray)
pyplot.title('CG')
#     pyplot.legend([im1, im im4])


pyplot.show()


