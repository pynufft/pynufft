#import pysap
import numpy as np
import matplotlib.pyplot as plt
#from pysap.data import get_sample_data

from pynufft import NUFFT_hsa
from pynufft import NUFFT_cpu
import numpy
import pkg_resources

DATA_PATH = pkg_resources.resource_filename('pynufft', './src/data/')   

single_image = numpy.load(DATA_PATH +'phantom_3D_128_128_128.npz')['arr_0']
# Load 3D images the data
#Il = get_sample_data("3d-pmri")

# Cropping the images to be [128, 128, 128] 
#Il.data = Il.data[:, :, :, 15:128+15]

# Taking the 3 first coils sensinsities
#Il = Il.data[:3]
# single_image = np.load('phantom_3D_128_128_128.npz')['arr_0']
x = np.broadcast_to(np.reshape(single_image, (128,128,128,1)), (128,128,128,4)).copy()

x[0:64,:,:,0] = x[0:64,:,:,0]*0.5
x[60:70,:,:,1] = x[60:70,:,:,1]*0.5
x[64:,:,:,2] = x[64:,:,:,2]*0.5

Il = x
# Check the shape of the images
print("Images shapes: ", x.shape)

# Create a full-kspace sampling scheme
def convert_mask_to_locations_3D(mask):
    """ Return the converted Cartesian mask as sampling locations.
    Parameters
    ----------
    mask: np.ndarray, {0,1}
        2D matrix, not necessarly a square matrix.
    Returns
    -------
    samples_locations: np.ndarray
        list of the samples between [-0.5, 0.5[.
    """
    dim1, dim2, dim3 = np.where(mask == 1)
    dim1 = dim1.astype("float") / mask.shape[0] - 0.5
    dim2 = dim2.astype("float") / mask.shape[1] - 0.5
    dim3 = dim3.astype("float") / mask.shape[2] - 0.5
    return np.c_[dim1, dim2, dim3]

#samples = convert_mask_to_locations_3D(np.ones(Il.shape[0:3]))
#samples = 2 * np.pi * samples

samples = np.random.randn(2085640,3)*3.1415

# Create a NUFFT object 
nufftObj = NUFFT_hsa(API='ocl',
                     platform_number=1,
                     device_number=0,
                     verbosity=0)
nufftObj.plan(om=samples,
              Nd=(128,128,128),
              Kd=tuple([256, 256, 256]),
              Jd=tuple([5, 5, 5]),
              batch=4,
              ft_axes=(0,1,2),
              radix=1)

# Casting the dtype to be complex64
dtype = np.complex64

# Computing the forward pass of the NUFFT

# JML: not needed in the new version
# nufftObj.x_Nd = nufftObj.thr.to_device(Il[:3].astype(dtype))
# gx = nufftObj.thr.copy_array(nufftObj.x_Nd)

#x = numpy.einsum('cxyz -> xyzc', Il[0:3]).copy() # coil must be the last dimension; Assume it is a C-order array



gy = nufftObj.forward(x)
#y = np.copy(gy.get())

# Checking the shape of the kspace
#print("K-space shapes: ", y.shape)

# Computing the backward pass

gx2 = nufftObj.adjoint(gy)
rec_Il = gx2.get() #np.copy(gx.get())
print(rec_Il.shape)
# Plotting the results
import matplotlib.pyplot
matplotlib.pyplot.subplot(2,3,1)
matplotlib.pyplot.imshow(x[:,:,64,0].real)
matplotlib.pyplot.title('input_image (64-th slice, first coil)')
matplotlib.pyplot.subplot(2,3,2)
matplotlib.pyplot.imshow(x[:,:,64,1].real)
matplotlib.pyplot.title('input_image (64-th slice, second coil)')
matplotlib.pyplot.subplot(2,3,3)
matplotlib.pyplot.imshow(x[:,:,64,2].real)
matplotlib.pyplot.title('input_image (64-th slice, third coil)')



matplotlib.pyplot.subplot(2,3,4)
matplotlib.pyplot.imshow(rec_Il[:,:,64,0].real)
matplotlib.pyplot.title('output_image (64-th slice, first coil)')
matplotlib.pyplot.subplot(2,3,5)
matplotlib.pyplot.imshow(rec_Il[:,:,64,1].real)
matplotlib.pyplot.title('output_image (64-th slice, second coil)')
matplotlib.pyplot.subplot(2,3,6)
matplotlib.pyplot.imshow(rec_Il[:,:,64,2].real)
matplotlib.pyplot.title('output_image (64-th slice, third coil)')

matplotlib.pyplot.show()
