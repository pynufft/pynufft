"""
NUFFT CPU class
=======================================
"""


from __future__ import absolute_import
import numpy
import scipy.sparse
import numpy.fft
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
    omN = numpy.zeros((N, dim), dtype = numpy.float64)
    grid = numpy.indices(Nd)    
    for dimid in range(0, dim):
        omN[:, dimid] = (grid[dimid].ravel() - Nd[dimid]/2 )
    M = om.shape[0]
    A = numpy.einsum('m, n -> mn', om[:, 0], omN[:,0], optimize='optimal')
    for d in range(1, dim):
        A += numpy.einsum('m, n -> mn', om[:, d], omN[:,d], optimize='optimal')
        
    return numpy.exp(-1.0j* A)   

class NUDFT_cpu:
    """
    Class NUDFT_cpu
   """
    def __init__(self):
        """
        Constructor.

        :param None:
        :type None: Python NoneType
        :return: NUFFT: the pynufft_hsa.NUFFT instance
        :rtype: NUFFT: the pynufft_hsa.NUFFT class
        :Example:

        >>> from pynufft import NUFFT_cpu
        >>> NufftObj = NUFFT_cpu()
        """
        self.dtype = numpy.complex64  # : initial value: numpy.complex64
        self.debug = 0  #: initial value: 0
        self.Nd = ()  # : initial value: ()
#         self.Kd = ()  # : initial value: ()
#         self.Jd = ()  #: initial value: ()
        self.ndims = 0  # : initial value: 0
#         self.ft_axes = ()  # : initial value: ()
        self.batch = None  # : initial value: None
        pass
    def plan(self, om, Nd):
        self.Nd = Nd
        self.ndims = len(Nd)
        self.F_matrix = []
        compute_str_x = ''
        F_str = ''
        for dimid in range(0, self.ndims):
            self.F_matrix += [DFT_matrix((Nd[dimid],), om[:, dimid:dimid+1]),]
            compute_str_x += lower_case[dimid]
            F_str += ', m' + lower_case[dimid]
        self.compute_str_forward = compute_str_x + F_str + '-> m'
        self.compute_str_adj = 'm' + F_str + '->' + compute_str_x 
        
        self.scale = numpy.prod(Nd)
    def forward(self, x):
        
        y = numpy.einsum(self.compute_str_forward, x, *self.F_matrix)
        return y
    def adjoint(self, y):
        x = numpy.einsum(self.compute_str_adj, y, *[self.F_matrix[dimid].conj() for dimid in range(0, self.ndims)])
        x /= self.scale
        return x.reshape(self.Nd)