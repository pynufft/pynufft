"""
NUFFT class
=======================================
"""
from __future__ import absolute_import
import numpy
import warnings
import scipy.sparse
import numpy.fft
#import scipy.signal
import scipy.linalg
import scipy.special
from functools import wraps as _wraps
# from ..linalg.nufft_cpu import NUFFT_cpu
# from ..linalg.nufft_hsa import NUFFT_hsa





def push_cuda_context(hsa_method):
    """
    Decorator: Push cude context to the top of the stack for current use
    Add @push_cuda_context before the methods of NUFFT_device()
    """
    @_wraps(hsa_method)
    def wrapper(*args, **kwargs):
        try:
            args[0].thr._context.push()
        except:
            pass
        return hsa_method(*args, **kwargs)
    return wrapper



class NUFFT:
    """
    NUFFT class
    =======================================
    A super class of cpu and gpu NUFFT functions. 
    
    Note: NUFFT does not inherit NUFFT_cpu (deprecated) and NUFFT_hsa (deprecated).
    """
    #import cpu codes
    from ._nufft_class_methods_cpu import _init__cpu, _plan_cpu, _precompute_sp_cpu, _solve_cpu, _forward_cpu, _adjoint_cpu, _selfadjoint_cpu, _selfadjoint2_cpu, _x2xx_cpu, _xx2k_cpu, _xx2k_one2one_cpu, _k2vec_cpu, _vec2y_cpu, _k2y_cpu, _y2vec_cpu, _vec2k_cpu, _y2k_cpu, _k2xx_cpu, _k2xx_one2one_cpu, _xx2x_cpu, _k2y2k_cpu
    # import host codes
    from ._nufft_class_methods_cpu import  _forward_host,  _adjoint_host, _selfadjoint_host,  _solve_host, _xx2k_host, _k2xx_host, _x2xx_host, _xx2x_host, _k2y_host,  _y2k_host
    # import device codes
    from ._nufft_class_methods_device import _init__device, _plan_device,  _set_wavefront_device, _offload_device, to_device, to_host, _x2xx_device, _xx2k_device, _k2y_device, _y2k_device, _k2xx_device, _xx2x_device,  _selfadjoint_device, _forward_device,  _adjoint_device, release, _solve_device, release
    
    # legacy codes (csr format for device)
    from ._nufft_class_methods_device import _y2k_legacy, _k2y_legacy, _forward_legacy, _adjoint_legacy, _selfadjoint_legacy, _plan_legacy, _offload_legacy, _solve_legacy
    from ._nufft_class_methods_cpu import _k2y_legacy_host,  _y2k_legacy_host, _selfadjoint_legacy_host, _forward_legacy_host, _adjoint_legacy_host,  _solve_legacy_host
    
    def __init__(self, device_indx=None, legacy=None):
        """
        Constructor.

        :param None:
        :type None: Python NoneType
        :return: NUFFT: the pynufft.NUFFT instance
        :rtype: NUFFT: the pynufft.NUFFT class
        :Example:

        >>> from pynufft import NUFFT
        >>> NufftObj = NUFFT()
        
        or 
        
        >>> from pynufft import NUFFT, helper
        >>> device = helper.device_list()[0]
        >>> NufftObj = NUFFT(device) # for first acceleration device in the system
        
        """
        if device_indx is None:
            self._init__cpu()
            self.processor = 'cpu'
        else:
            if legacy is True:
                self._init__device(device_indx)
                self.processor = 'hsa_legacy'
            else:
                self._init__device(device_indx)
                self.processor = 'hsa'
            
    def __del__(self):
        if self.processor is 'hsa' or 'hsa_legacy':
            self.release()
        else:
            pass
                
    def plan(self,  *args, **kwargs):
        """
        Plan the NUFFT object with the geometry provided.

        :param om: The M off-grid locates in the frequency domain,
                    which is normalized between [-pi, pi]
        :param Nd: The matrix size of the equispaced image.
                   Example: Nd=(256,256) for a 2D image;
                             Nd = (128,128,128) for a 3D image
        :param Kd: The matrix size of the oversampled frequency grid.
                   Example: Kd=(512,512) for 2D image;
                            Kd = (256,256,256) for a 3D image
        :param Jd: The interpolator size.
                   Example: Jd=(6,6) for 2D image;
                            Jd = (6,6,6) for a 3D image
        :param ft_axes: (Optional) The axes for Fourier transform.
                        The default is all axes if 'None' is given.
        :type om: numpy.float array, matrix size = M * ndims
        :type Nd: tuple, ndims integer elements.
        :type Kd: tuple, ndims integer elements.
        :type Jd: tuple, ndims integer elements.
        :type ft_axes: None, or tuple with optional integer elements.
        :returns: 0
        :rtype: int, float

        :ivar Nd: initial value: Nd
        :ivar Kd: initial value: Kd
        :ivar Jd: initial value: Jd
        :ivar ft_axes: initial value: None

        :Example:

        >>> from pynufft import NUFFT
        >>> NufftObj = NUFFT()
        >>> NufftObj.plan(om, Nd, Kd, Jd)

        or

        >>> NufftObj.plan(om, Nd, Kd, Jd, ft_axes)

        """        
        func = {'cpu': self._plan_cpu,
                    'hsa': self._plan_device,
                    'hsa_legacy': self._plan_legacy}
        return func.get(self.processor)(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """
        Forward NUFFT (host code)

        :param x: The input numpy array, with the size of Nd 
        :type: numpy array with the dtype of numpy.complex64
        :return: y: The output numpy array, with the size of (M,) 
        :rtype: numpy array with the dtype of numpy.complex64
        """        
        func = {'cpu': self._forward_cpu, 
                    'hsa': self._forward_host,
                    'hsa_legacy': self._forward_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def adjoint(self, *args, **kwargs):
        """
        Adjoint NUFFT (host code)

        :param y: The input numpy array, with the size of (M,) 
        :type: numpy array with the dtype of numpy.complex64
        :return: x: The output numpy array,
                    with the size of Nd or Nd 
        :rtype: numpy array with the dtype of numpy.complex64
        """        
        func = {'cpu': self._adjoint_cpu,
                'hsa': self._adjoint_host,
                'hsa_legacy': self._adjoint_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def selfadjoint(self, *args, **kwargs):
        """
        selfadjoint NUFFT (host code)

        :param x: The input numpy array, with size=Nd
        :type: numpy array with dtype =numpy.complex64
        :return: x: The output numpy array, with size=Nd
        :rtype: numpy array with dtype =numpy.complex64
        """
        func = {'cpu': self._selfadjoint_cpu,
                'hsa': self._selfadjoint_host,
                'hsa_legacy': self._selfadjoint_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def solve(self, *args, **kwargs):
        """
        Solve NUFFT (host code)
        :param y: data, numpy.complex64. The shape = (M,) 
        :param solver: 'cg', 'L1TVOLS', 'lsmr', 'lsqr', 'dc', 'bicg',
                       'bicgstab', 'cg', 'gmres','lgmres'
        :param maxiter: the number of iterations
        :type y: numpy array, dtype = numpy.complex64
        :type solver: string
        :type maxiter: int
        :return: numpy array with size Nd.
        """        
        func = {'cpu': self._solve_cpu,
                'hsa': self._solve_host,
                'hsa_legacy': self._solve_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def xx2k(self, *args, **kwargs):
        func = {'cpu': self._xx2k_cpu,
                'hsa': self._xx2k_host,
                'hsa_legacy': self._xx2k_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def k2xx(self, *args, **kwargs):
        func = {'cpu': self._k2xx_cpu,
                'hsa': self._k2xx_host,
                'hsa_legacy': self._k2xx_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def x2xx(self, *args, **kwargs):
        func = {'cpu': self._x2xx_cpu,
                'hsa': self._x2xx_host,
                'hsa_legacy': self._x2xx_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def xx2x(self, *args, **kwargs):
        func = {'cpu': self._xx2x_cpu,
                'hsa': self._xx2x_host,
                'hsa_legacy': self._xx2x_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def k2y(self, *args, **kwargs):
        func = {'cpu': self._k2y_cpu,
                'hsa': self._k2y_host,
                'hsa_legacy': self._k2y_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def y2k(self, *args, **kwargs):
        func = {'cpu': self._y2k_cpu,
                'hsa': self._y2k_host,
                'hsa_legacy': self._y2k_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def k2yk2(self, *args, **kwargs):
        func = {'cpu': self._k2yk2_cpu,
                'hsa': self._k2yk2_host,
                'hsa_legacy': self._k2yk2_host}
        return func.get(self.processor)(*args, **kwargs)
    
#     def adjoint_many2one(self, *args, **kwargs):
#         func = {'cpu': self._adjoint_many2one_cpu,
#                 'hsa': self._adjoint_many2one_host,
#                 'hsa_legacy': self._adjoint_many2one_legacy_host}
#         return func.get(self.processor)(*args, **kwargs)
#     
#     def forward_one2many(self, *args, **kwargs):
#         func = {'cpu': self._forward_one2many_cpu,
#                 'hsa': self._forward_one2many_host,
#                 'hsa_legacy': self._forward_one2many_legacy_host}
#         return func.get(self.processor)(*args, **kwargs)
    
#     def selfadjoint_one2many2one(self, *args, **kwargs):
#         func = {'cpu': self._selfadjoint_one2many2one_cpu,
#                 'hsa': self._selfadjoint_one2many2one_host,
#                 'hsa_legacy': self._selfadjoint_one2many2one_legacy_host}
#         return func.get(self.processor)(*args, **kwargs)   
    
    def k2xx_one2one(self, *args, **kwargs):
        func = {'cpu':self._k2xx_one2one_cpu}
        return func.get(self.processor)(*args, **kwargs)
    
    def xx2k_one2one(self, *args, **kwargs):
        func = {'cpu':self._xx2k_one2one_cpu}
        return func.get(self.processor)(*args, **kwargs) 
    
    def k2y2k(self, *args, **kwargs):
        func = {'cpu': self._k2y2k_cpu}
        return func.get(self.processor)(*args, **kwargs)
#     def set_sense(self, *args, **kwargs):
#         func = {'cpu': self._set_sense_cpu,
#                 'hsa': self._set_sense_host,
#                 'hsa_legacy': self._set_sense_host}
#         return func.get(self.processor)(*args, **kwargs)
#     def reset_sense(self, *args, **kwargs):
#         func = {'cpu': self._reset_sense_cpu,
#                 'hsa': self._reset_sense_host,
#                 'hsa_legacy': self._reset_sense_host}
#         return func.get(self.processor)(*args, **kwargs)
    
