"""
NUDFT class
=======================================
"""


from __future__ import absolute_import
import cupy 
import numpy 
import scipy.sparse
# import cupy.fft as numpy.fft
# import scipy.signal
import scipy.linalg
# import scipy.special

# from ..src._helper import helper, helper1
lower_case = 'abcdefghijklnopqrstuvwxyABCDEFGHIJKLNOPQRSTUVWXY'

def fake_Cartesian(Nd):
    dim = len(Nd) # dimension
    M = numpy.prod(Nd)
    om = numpy.zeros((M, dim), dtype = numpy.float64)
    grid = numpy.indices(Nd)
    for dimid in range(0, dim):
        om[:, dimid] = (grid[dimid].ravel() *2/ Nd[dimid] - 1.0)*numpy.pi
    return om    

def DFT_matrix(Nd, om=None):
    dim = len(Nd) # dimension
    if om is None:
        om = fake_Cartesian(Nd)
    N = numpy.prod(Nd)
    omN = cupy.zeros((N, dim), dtype = numpy.float64)
    grid = cupy.indices(Nd)    
    for dimid in range(0, dim):
        omN[:, dimid] = (grid[dimid].ravel() - Nd[dimid]/2 )
    M = om.shape[0]
    A = cupy.einsum('m, n -> mn', om[:, 0], omN[:,0], optimize='optimal')
    for d in range(1, dim):
        A += cupy.einsum('m, n -> mn', om[:, d], omN[:,d], optimize='optimal')
        
    return cupy.exp(-1.0j* A)   

class NUDFT_cupy:
    """
    Class NUDFT
    =============================
    The non-uniform DFT operator
   """
    def __init__(self):
        """
        Constructor.

        :param None:
        :type None: Python NoneType
        :return: NUFFT: the pynufft.NUDFT instance
        :rtype: NUFFT: the pynufft.NUFFT class
        :Example:

        >>> from pynufft import NUDFT
        >>> NufftObj = NUDFT()
        """
        self.dtype = numpy.complex128  # : initial value: numpy.complex64
        self.debug = 0  #: initial value: 0
        self.Nd = ()  # : initial value: ()
#         self.Kd = ()  # : initial value: ()
#         self.Jd = ()  #: initial value: ()
        self.ndims = 0  # : initial value: 0
#         self.ft_axes = ()  # : initial value: ()
        self.batch = None  # : initial value: None
        pass
    def plan(self, om, Nd):
#         if batch != None:
#             self.batch = numpy.int(batch)
        self.Nd = Nd
        self.ndims = len(Nd)
        self.F_mats = []
        compute_str_x = ''
        F_str = ''
        for dimid in range(0, self.ndims):
            self.F_mats += [DFT_matrix((Nd[dimid],), om[:, dimid:dimid+1]),]
            compute_str_x += lower_case[dimid]
            F_str += ', m' + lower_case[dimid]
        
#         if type(batch)==numpy.int:
#             self.compute_str_forward = compute_str_x +'z' + F_str + '-> mz'
#             self.compute_str_adj = 'm' + F_str + '->' + compute_str_x +'z'
#         else: 
        self.compute_str_forward = compute_str_x + F_str + '-> m'
        self.compute_str_adj = 'm' + F_str + '->' + compute_str_x
        
        self.scale = numpy.prod(Nd)
    def forward(self, x):
        try:
            y = cupy.einsum(self.compute_str_forward, x, *self.F_mats,optimize='optimal')
        except:
#             self.path = cupy.einsum_path(self.compute_str_forward, x, *self.F_mats,optimize='optimal')[0]
            y = cupy.einsum(self.compute_str_forward, x, *self.F_mats,optimize='optimal')
        return y
    def adjoint(self, y):
#         print(self.compute_str_adj)
        x = cupy.einsum(self.compute_str_adj, y, *[self.F_mats[dimid].conj() for dimid in range(0, self.ndims)],optimize='optimal')
        x /= self.scale
        return x.reshape(self.Nd)
