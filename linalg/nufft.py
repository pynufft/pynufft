"""
NUFFT HSA classes
=======================================

"""
from __future__ import absolute_import
import numpy
import warnings
import scipy.sparse
import numpy.fft
import scipy.signal
import scipy.linalg
import scipy.special
from functools import wraps as _wraps
# from ..linalg.nufft_cpu import NUFFT_cpu
# from ..linalg.nufft_hsa import NUFFT_hsa

from ..src._helper import helper, helper1



def push_cuda_context(hsa_method):
    """
    Decorator: Push cude context to the top of the stack for current use
    Add @push_cuda_context before the methods of NUFFT_hsa()
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
    A super class of cpu and gpu NUFFT functions. 
    Note: NUFFT does NOT inherit NUFFT_cpu and NUFFT_hsa.
    Multi-coil or single-coil memory reduced NUFFT.

    """
    def __init__(self, device_indx=None):
        if device_indx is None:
            self.__init__cpu()
            self.processor = 'cpu'
        else:
            self.__init__hsa(device_indx)
            self.processor = 'hsa'
            
    def __init__cpu(self):
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
            self.Kd = ()  # : initial value: ()
            self.Jd = ()  #: initial value: ()
            self.ndims = 0  # : initial value: 0
            self.ft_axes = ()  # : initial value: ()
            self.batch = None  # : initial value: None
            pass    
    def __init__hsa(self, device_indx=None):
        """
        Constructor.

        :param API: The API for the heterogeneous system. API='cuda'
                    or API='ocl'
        :param platform_number: The number of the platform found by the API.
        :param device_number: The number of the device found on the platform.
        :param verbosity: Defines the verbosity level, default value is 0
        :type API: string
        :type platform_number: integer
        :type device_number: integer
        :type verbosity: integer
        :returns: 0
        :rtype: int, float

        :Example:

        >>> from pynufft import NUFFT_hsa
        >>> NufftObj = NUFFT_hsa(API='cuda', platform_number=0,
                                         device_number=0, verbosity=0)
        """

        self.dtype = numpy.complex64
        self.verbosity = 0#verbosity

        from reikna import cluda
        import reikna.transformations
        from reikna.cluda import functions, dtypes
#         try:  # try to create api/platform/device using the given parameters
        API = device_indx[0]
        
#             if 'cuda' == API:
#                 api = cluda.cuda_api()
#             elif 'ocl' == API:
#                 api = cluda.ocl_api()
#         api = device_indx[0]
        platform_number = device_indx[1]
        device_number = device_indx[2]
        
#         NUFFT_hsa.__init__(self, API, platform_number, device_number)
        
        platform = device_indx[3]
        device = device_indx[4]
#             self.thr = api.Thread(device)
        self.device = device
        self.thr = device_indx[5]
#             api = device_indx[6]
#             self.thr = api.Thread(device)  # pyopencl.create_some_context()
#         self.device = device  # : device name

#         """
#         Wavefront: as warp in cuda. Can control the width in a workgroup
#         Wavefront is required in spmv_vector as it improves data coalescence.
#         see cCSR_spmv and zSparseMatVec
#         """
        self.wavefront = device_indx[6]
#         if self.verbosity > 0:
#             print('Wavefront of OpenCL (as wrap of CUDA) = ', self.wavefront)
#             print('API = ', API)
#             print('thr = ',  self.thr
#                   )
        from ..src import re_subroutine  # import create_kernel_sets
        kernel_sets = re_subroutine.create_kernel_sets(API)

        prg = self.thr.compile(kernel_sets,
                               render_kwds=dict(LL=str(self.wavefront)),
                               fast_math=False)
        self.prg = prg
        self.processor = 'hsa'
        print('self is?', self.thr)
    def _set_wavefront_hsa(self, wf):
        try:
            self.wavefront = int(wt)#api.DeviceParameters(device).warp_size
            if self.verbosity > 0:
                print('Wavefront of OpenCL (as wrap of CUDA) = ', self.wavefront)
    
            from ..src import re_subroutine  # import create_kernel_sets
            kernel_sets = re_subroutine.create_kernel_sets(API)
    
            prg = self.thr.compile(kernel_sets,
                                   render_kwds=dict(LL=str(self.wavefront)),
                                   fast_math=False)
            self.prg = prg
        except:
            print('Failled at set_wavefront found')
    def plan(self,  *args, **kwargs):
        func = {'cpu': self._plan_cpu,
                    'hsa': self._plan_hsa}
        return func.get(self.processor)(*args, **kwargs)
    def forward(self, *args, **kwargs):
        func = {'cpu': self._forward_cpu, 
                    'hsa': self._forward_hsa}
        return func.get(self.processor)(*args, **kwargs)
    def adjoint(self, *args, **kwargs):
        func = {'cpu': self._adjoint_cpu,
                'hsa': self._adjoint_hsa}
        return func.get(self.processor)(*args, **kwargs)
    def selfadjoint(self, *args, **kwargs):
        func = {'cpu': self._selfadjoint_cpu,
                'hsa': self._selfadjoint_hsa}
        return func.get(self.processor)(*args, **kwargs)
    def solve(self, *args, **kwargs):
        func = {'cpu': self._solve_cpu,
                'hsa': self._solve_hsa}
        return func.get(self.processor)(*args, **kwargs)
    
    def xx2k(self, *args, **kwargs):
        func = {'cpu': self._xx2k_cpu,
                'hsa': self._xx2k_hsa}
        return func.get(self.processor)(*args, **kwargs)
    
    def k2xx(self, *args, **kwargs):
        func = {'cpu': self._k2xx_cpu,
                'hsa': self._k2xx_hsa}
        return func.get(self.processor)(*args, **kwargs)
    
    def x2xx(self, *args, **kwargs):
        func = {'cpu': self._x2xx_cpu,
                'hsa': self._x2xx_hsa}
        return func.get(self.processor)(*args, **kwargs)
    
    def x2xx(self, *args, **kwargs):
        func = {'cpu': self._x2xx_cpu,
                'hsa': self._x2xx_hsa}
        return func.get(self.processor)(*args, **kwargs)
    
    def k2y(self, *args, **kwargs):
        func = {'cpu': self._k2y_cpu,
                'hsa': self._k2y_hsa}
        return func.get(self.processor)(*args, **kwargs)
    def y2k(self, *args, **kwargs):
        func = {'cpu': self._y2k_cpu,
                'hsa': self._y2k_hsa}
        return func.get(self.processor)(*args, **kwargs)
    
    def k2yk2(self, *args, **kwargs):
        func = {'cpu': self._k2yk2_cpu,
                'hsa': self._k2yk2_hsa}
        return func.get(self.processor)(*args, **kwargs)
    
    def adjoint_many2one(self, *args, **kwargs):
        func = {'cpu': self._adjoint_many2one_cpu,
                'hsa': self._adjoint_many2one_hsa}
        return func.get(self.processor)(*args, **kwargs)
    
    def forward_one2many(self, *args, **kwargs):
        func = {'cpu': self._forward_one2many_cpu,
                'hsa': self._forward_one2many_hsa}
        return func.get(self.processor)(*args, **kwargs)
    
    def selfadjoint_one2many2one(self, *args, **kwargs):
        func = {'cpu': self._selfadjoint_one2many2one_cpu,
                'hsa': self._selfadjoint_one2many2one_hsa}
        return func.get(self.processor)(*args, **kwargs)
    def _plan_cpu(self, om, Nd, Kd, Jd, ft_axes=None, batch=None):
        """
        Plan the NUFFT_cpu object with the geometry provided.

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
        :param batch: (Optional) Batch mode.
                     If the batch is provided, the last appended axis is the number
                     of identical NUFFT to be transformed.
                     The default is 'None'.
        :type om: numpy.float array, matrix size = M * ndims
        :type Nd: tuple, ndims integer elements.
        :type Kd: tuple, ndims integer elements.
        :type Jd: tuple, ndims integer elements.
        :type ft_axes: None, or tuple with optional integer elements.
        :type batch: None, or integer
        :returns: 0
        :rtype: int, float

        :ivar Nd: initial value: Nd
        :ivar Kd: initial value: Kd
        :ivar Jd: initial value: Jd
        :ivar ft_axes: initial value: None
        :ivar batch: initial value: None

        :Example:

        >>> from pynufft import NUFFT_cpu
        >>> NufftObj = NUFFT_cpu()
        >>> NufftObj.plan(om, Nd, Kd, Jd)

        or

        >>> NufftObj.plan(om, Nd, Kd, Jd, ft_axes, batch)

        """

        self.ndims = len(Nd)  # : initial value: len(Nd)
        if ft_axes is None:
            ft_axes = range(0, self.ndims)
        self.ft_axes = ft_axes  # default: all axes (range(0, self.ndims)

        self.st = helper.plan(om, Nd, Kd, Jd, ft_axes=ft_axes,
                              format='CSR')

        self.Nd = self.st['Nd']  # backup
        self.Kd = self.st['Kd']
        # backup
        self.sn = numpy.asarray(self.st['sn'].astype(self.dtype), order='C')

        if batch is None:  # single-coil
            self.parallel_flag = 0
            self.batch = 1

        else:  # multi-coil
            self.parallel_flag = 1
            self.batch = batch

        if self.parallel_flag is 1:
            self.multi_Nd = self.Nd + (self.batch, )
            self.uni_Nd = self.Nd + (1, )
            self.multi_Kd = self.Kd + (self.batch, )
            self.multi_M = (self.st['M'], ) + (self.batch, )
            self.multi_prodKd = (numpy.prod(self.Kd), self.batch)
            self.sn = numpy.reshape(self.sn, self.Nd + (1,))

        elif self.parallel_flag is 0:
            self.multi_Nd = self.Nd  # + (self.Reps, )
            self.uni_Nd = self.Nd
            self.multi_Kd = self.Kd  # + (self.Reps, )
            self.multi_M = (self.st['M'], )
            self.multi_prodKd = (numpy.prod(self.Kd), )

        # Calculate the density compensation function
        self.sp = self.st['p'].copy().tocsr()
        
        self.spH = (self.st['p'].getH().copy()).tocsr()
        
        self.Kdprod = numpy.int32(numpy.prod(self.st['Kd']))
        self.Jdprod = numpy.int32(numpy.prod(self.st['Jd']))
        del self.st['p'], self.st['sn']

        self.NdCPUorder, self.KdCPUorder, self.nelem = helper.preindex_copy(
            self.st['Nd'],
            self.st['Kd'])
        self.volume = {}
        self.volume['cpu_coil_profile'] = numpy.ones(self.multi_Nd)

        return 0

    def _plan_hsa(self, om, Nd, Kd, Jd, ft_axes=None, batch=None, radix=None):
        """
        Design the multi-coil or single-coil memory reduced interpolator.


        :param om: The M off-grid locations in the frequency domain.
                   Normalized between [-pi, pi]
        :param Nd: The matrix size of equispaced image.
                    Example: Nd=(256, 256) for a 2D image;
                             Nd = (128, 128, 128) for a 3D image
        :param Kd: The matrix size of the oversampled frequency grid.
                   Example: Kd=(512,512) for 2D image;
                   Kd = (256,256,256) for a 3D image
        :param Jd: The interpolator size.
                   Example: Jd=(6,6) for 2D image;
                   Jd = (6,6,6) for a 3D image
        :param ft_axes: The dimensions to be transformed by FFT.
                   Example: ft_axes = (0, 1) for 2D,
                   ft_axes = (0, 1, 2) for 3D;
                   ft_axes = None for all dimensions.
        :param batch: Batch NUFFT.
                    If provided, the shape is Nd + (batch, ).
                    The last axis is the number of parallel coils.
                    batch = None for single coil.
        :param radix: ????.
                    If provided, the shape is Nd + (batch, ).
                    The last axis is the number of parallel coils.
                    batch = None for single coil.
        :type om: numpy.float array, matrix size = (M, ndims)
        :type Nd: tuple, ndims integer elements.
        :type Kd: tuple, ndims integer elements.
        :type Jd: tuple, ndims integer elements.
        :type ft_axes: tuple, selected axes to be transformed.
        :type batch: int or None
        :returns: 0
        :rtype: int, float
        :Example:

        >>> from pynufft import NUFFT_hsa
        >>> NufftObj = NUFFT_hsa()
        >>> NufftObj.plan(om, Nd, Kd, Jd)

        """

        self.ndims = len(Nd)  # dimension
        if ft_axes is None:
            ft_axes = range(0, self.ndims)
        self.ft_axes = ft_axes

        self.st = helper.plan(om, Nd, Kd, Jd, ft_axes=ft_axes,
                              format='pELL', radix=radix)
        if batch is None:
            self.parallel_flag = 0
        else:
            self.parallel_flag = 1

        if batch is None:
            self.batch = numpy.uint32(1)

        else:
            self.batch = numpy.uint32(batch)

        self.Nd = self.st['Nd']  # backup
        self.Kd = self.st['Kd']
        #  self.sn = numpy.asarray(self.st['sn'].astype(self.dtype),
        #                            order='C')# backup
        if self.batch == 1 and (self.parallel_flag == 0):
            self.multi_Nd = self.Nd
            self.multi_Kd = self.Kd
            self.multi_M = (self.st['M'], )
            # Broadcasting the sense and scaling factor (Roll-off)
            # self.sense2 = self.sense*numpy.reshape(self.sn, self.Nd + (1, ))
        else:  # self.batch is 0:
            self.multi_Nd = self.Nd + (self.batch, )
            self.multi_Kd = self.Kd + (self.batch, )
            self.multi_M = (self.st['M'], ) + (self.batch, )
        self.invbatch = 1.0 / self.batch
        self.Kdprod = numpy.uint32(numpy.prod(self.st['Kd']))
        self.Jdprod = numpy.uint32(numpy.prod(self.st['Jd']))
        self.Ndprod = numpy.uint32(numpy.prod(self.st['Nd']))

        self.Nd_elements, self.invNd_elements = helper.strides_divide_itemsize(
                                                    self.st['Nd'])
        # only return the Kd_elements
        self.Kd_elements = helper.strides_divide_itemsize(self.st['Kd'])[0]
        self.NdCPUorder, self.KdCPUorder, self.nelem = helper.preindex_copy(
                                                        self.st['Nd'],
                                                        self.st['Kd'])
        self._offload_hsa()

        return 0

    @push_cuda_context
    def _offload_hsa(self):  # API, platform_number=0, device_number=0):
        """
        self.offload():

        Off-load NUFFT to the opencl or cuda device(s)

        :param API: define the device type, which can be 'cuda' or 'ocl'
        :param platform_number: define which platform to be used.
                                The default platform_number = 0.
        :param device_number: define which device to be used.
                            The default device_number = 0.
        :type API: string
        :type platform_number: int
        :type device_number: int
        :return: self: instance
        """

        self.pELL = {}  # dictionary

        self.pELL['nRow'] = numpy.uint32(self.st['pELL'].nRow)
        self.pELL['prodJd'] = numpy.uint32(self.st['pELL'].prodJd)
        self.pELL['sumJd'] = numpy.uint32(self.st['pELL'].sumJd)
        self.pELL['dim'] = numpy.uint32(self.st['pELL'].dim)
        self.pELL['Jd'] = self.thr.to_device(
            self.st['pELL'].Jd.astype(numpy.uint32))
        self.pELL['meshindex'] = self.thr.to_device(
            self.st['pELL'].meshindex.astype(numpy.uint32))
        self.pELL['kindx'] = self.thr.to_device(
            self.st['pELL'].kindx.astype(numpy.uint32))
        self.pELL['udata'] = self.thr.to_device(
            self.st['pELL'].udata.astype(self.dtype))

        self.volume = {}

        self.volume['Nd_elements'] = self.thr.to_device(
            numpy.asarray(self.Nd_elements, dtype=numpy.uint32))
        self.volume['Kd_elements'] = self.thr.to_device(
            numpy.asarray(self.Kd_elements, dtype=numpy.uint32))
        self.volume['invNd_elements'] = self.thr.to_device(
            self.invNd_elements.astype(numpy.float32))
        self.volume['Nd'] = self.thr.to_device(numpy.asarray(
            self.st['Nd'], dtype=numpy.uint32))
        self.volume['NdGPUorder'] = self.thr.to_device(self.NdCPUorder)
        self.volume['KdGPUorder'] = self.thr.to_device(self.KdCPUorder)
        self.volume['gpu_coil_profile'] = self.thr.array(
            self.multi_Nd, dtype=self.dtype).fill(1.0)

        Nd = self.st['Nd']
        # tensor_sn = numpy.empty((numpy.sum(Nd), ), dtype=numpy.float32)
        #
        # shift = 0
        # for dimid in range(0, len(Nd)):
        #
        #     tensor_sn[shift :shift + Nd[dimid]] = \
        #     self.st['tensor_sn'][dimid][:, 0].real
        #     shift = shift + Nd[dimid]
        # self.volume['tensor_sn'] = self.thr.to_device(
        #     self.st['tensor_sn'].astype(numpy.float32))
        self.tSN = {}
        self.tSN['Td_elements'] = self.thr.to_device(
            numpy.asarray(self.st['tSN'].Td_elements, dtype=numpy.uint32))
        self.tSN['invTd_elements'] = self.thr.to_device(
            self.st['tSN'].invTd_elements.astype(numpy.float32))
        self.tSN['Td'] = self.thr.to_device(
            numpy.asarray(self.st['tSN'].Td, dtype=numpy.uint32))
        self.tSN['Tdims'] = self.st['tSN'].Tdims
        self.tSN['tensor_sn'] = self.thr.to_device(
            self.st['tSN'].tensor_sn.astype(numpy.float32))

        self.Ndprod = numpy.int32(numpy.prod(self.st['Nd']))
        self.Kdprod = numpy.int32(numpy.prod(self.st['Kd']))
        self.M = numpy.int32(self.st['M'])

        import reikna.fft
        if self.batch > 1:  # batch mode
            self.fft = reikna.fft.FFT(
                numpy.empty(self.st['Kd']+(self.batch, ), dtype=self.dtype),
                self.ft_axes).compile(self.thr, fast_math=False)
        else:  # elf.Reps == 1 Batch mode is wrong for
            self.fft = reikna.fft.FFT(
                numpy.empty(self.st['Kd'], dtype=self.dtype),
                self.ft_axes).compile(self.thr, fast_math=False)

        self.zero_scalar = self.dtype(0.0+0.0j)
        del self.st['pELL']
        if self.verbosity > 0:
            print('End of offload')

    def _precompute_sp_cpu(self):
        """

        Private: Precompute adjoint (gridding) and Toepitz interpolation
                 matrix.

        :param None:
        :type None: Python Nonetype
        :return: self: instance
        """
        try:
            W0 = numpy.ones((self.st['M'],), dtype=numpy.complex64)
            W = self.xx2k(self.adjoint(W0))
            self.W = (W*W.conj())**0.5
            del W0
            del W
        except:
            print("errors occur in self.precompute_sp()")
            raise

    def _reset_sense_cpu(self):
        self.volume['cpu_coil_profile'].fill(1.0)

    def _set_sense_cpu(self, coil_profile):

        self.volume = {}

        if coil_profile.shape == self.Nd + (self.batch, ):
            self.volume['cpu_coil_profile'] = coil_profile
        else:
            print('The shape of coil_profile might be wrong')
            print('coil_profile.shape = ', coil_profile.shape)
            print('shape of Nd + (batch, ) = ', self.Nd + (self.batch, ))

    def _forward_one2many_cpu(self, x):
        """
        Assume x.shape = self.Nd

        """

#         try:
        x2 = x.reshape(self.uni_Nd, order='C')*self.volume['cpu_coil_profile']
#         except:
#         x2 = x
        y2 = self.forward(x2)

        return y2

    def _adjoint_many2one_cpu(self, y):
        """
        Assume y.shape = self.multi_M
        """
        x2 = self.adjoint(y)
        x = x2*self.volume['cpu_coil_profile'].conj()
        try:
            x3 = numpy.mean(x, axis=self.ndims)
        except:
            x3 = x
        del x

        return x3

    def _solve_cpu(self, y, solver=None, *args, **kwargs):
        """
        Solve NUFFT_cpu.
        :param y: data, numpy.complex64. The shape = (M,) or (M, batch)
        :param solver: 'cg', 'L1TVOLS', 'lsmr', 'lsqr', 'dc', 'bicg',
                       'bicgstab', 'cg', 'gmres','lgmres'
        :param maxiter: the number of iterations
        :type y: numpy array, dtype = numpy.complex64
        :type solver: string
        :type maxiter: int
        :return: numpy array with size.
                The shape = Nd ('L1TVOLS') or  Nd + (batch,)
                ('lsmr', 'lsqr', 'dc','bicg','bicgstab','cg', 'gmres','lgmres')
        """
        from ..linalg.solve_cpu import solve
        x2 = solve(self,  y,  solver, *args, **kwargs)
        return x2  # solve(self,  y,  solver, *args, **kwargs)

    def _forward_cpu(self, x):
        """
        Forward NUFFT on CPU

        :param x: The input numpy array, with the size of Nd or Nd + (batch,)
        :type: numpy array with the dtype of numpy.complex64
        :return: y: The output numpy array, with the size of (M,) or (M, batch)
        :rtype: numpy array with the dtype of numpy.complex64
        """
        y = self._k2y_cpu(self._xx2k_cpu(self._x2xx_cpu(x)))

        return y

    def _adjoint_cpu(self, y):
        """
        Adjoint NUFFT on CPU

        :param y: The input numpy array, with the size of (M,) or (M, batch)
        :type: numpy array with the dtype of numpy.complex64
        :return: x: The output numpy array,
                    with the size of Nd or Nd + (batch, )
        :rtype: numpy array with the dtype of numpy.complex64
        """
        x = self._xx2x_cpu(self._k2xx_cpu(self._y2k_cpu(y)))

        return x

    def _selfadjoint_one2many2one_cpu(self, x):
        y2 = self._forward_one2many_cpu(x)
        x2 = self._adjoint_many2one_cpu(y2)
        del y2
        return x2

    def _selfadjoint_cpu(self, x):
        """
        selfadjoint NUFFT (Toeplitz) on CPU

        :param x: The input numpy array, with size=Nd
        :type: numpy array with dtype =numpy.complex64
        :return: x: The output numpy array, with size=Nd
        :rtype: numpy array with dtype =numpy.complex64
        """
        # x2 = self.adjoint(self.forward(x))

        x2 = self._xx2x_cpu(self._k2xx_cpu(self._k2y2k_cpu(self._xx2k_cpu(self._x2xx_cpu(x)))))

        return x2

    def _selfadjoint2_cpu(self, x):
        try:
            x2 = self._k2xx_cpu(self.W * self._xx2k_cpu(x))
        except:
            self._precompute_sp_cpu()
            x2 = self._k2xx_cpu(self.W * self._xx2k_cpu(x))
        return x2

    def _x2xx_cpu(self, x):
        """
        Private: Scaling on CPU
        Inplace multiplication of self.x_Nd by the scaling factor self.sn.
        """
        xx = x * self.sn
        return xx

    def _xx2k_cpu(self, xx):
        """
        Private: oversampled FFT on CPU

        Firstly, zeroing the self.k_Kd array
        Second, copy self.x_Nd array to self.k_Kd array by cSelect
        Third, inplace FFT
        """
        
        output_x = numpy.zeros(self.multi_Kd, dtype=self.dtype, order='C')

        for bat in range(0, self.batch):
            output_x.ravel()[self.KdCPUorder * self.batch + bat] = xx.ravel()[
                self.NdCPUorder * self.batch + bat]

        k = numpy.fft.fftn(output_x, axes=self.ft_axes)
        
        return k

    def _xx2k_one2one_cpu(self, xx):
        """
        Private: oversampled FFT on CPU

        First, zeroing the self.k_Kd array
        Second, copy self.x_Nd array to self.k_Kd array by cSelect
        Third, inplace FFT
        """
        
        output_x = numpy.zeros(self.st['Kd'], dtype=self.dtype, order='C')

        # for bat in range(0, self.batch):
        output_x.ravel()[self.KdCPUorder] = xx.ravel()[self.NdCPUorder]

        k = numpy.fft.fftn(output_x, axes=self.ft_axes)
        
        return k

    def _k2vec_cpu(self, k):
        k_vec = numpy.reshape(k, self.multi_prodKd, order='C')
        return k_vec

    def _vec2y_cpu(self, k_vec):
        '''
        gridding:
        '''
        y = self.sp.dot(k_vec)
        # y = self.st['ell'].spmv(k_vec)

        return y

    def _k2y_cpu(self, k):
        """
        Private: interpolation by the Sparse Matrix-Vector Multiplication
        """
        y = self._vec2y_cpu(self._k2vec_cpu(k))
        # numpy.reshape(self.st['p'].dot(Xk), (self.st['M'], ), order='F')
        return y

    def _y2vec_cpu(self, y):
        '''
       regridding non-uniform data (unsorted vector)
        '''
        # k_vec = self.st['p'].getH().dot(y)
        k_vec = self.spH.dot(y)
        # k_vec = self.st['ell'].spmvH(y)

        return k_vec

    def _vec2k_cpu(self, k_vec):
        '''
        Sorting the vector to k-spectrum Kd array
        '''
        k = numpy.reshape(k_vec, self.multi_Kd, order='C')

        return k

    def _y2k_cpu(self, y):
        """
        Private: gridding by the Sparse Matrix-Vector Multiplication
        """
        k_vec = self._y2vec_cpu(y)
        k = self._vec2k_cpu(k_vec)
        return k

    def _k2xx_cpu(self, k):
        """
        Private: the inverse FFT and image cropping (which is the reverse of
                 _xx2k() method)
        """
#         dd = numpy.size(self.Kd)

        k = numpy.fft.ifftn(k, axes=self.ft_axes)
        xx = numpy.zeros(self.multi_Nd, dtype=self.dtype, order='C')
        for bat in range(0, self.batch):
            xx.ravel()[self.NdCPUorder * self.batch + bat] = k.ravel()[
                self.KdCPUorder * self.batch + bat]
#         xx = xx[crop_slice_ind(self.Nd)]
        return xx

    def k2xx_one2one(self, *args, **kwargs):
        func = {'cpu':self._k2xx_one2one_cpu}
        return func.get(self.processor)(*args, **kwargs)
    def xx2k_one2one(self, *args, **kwargs):
        func = {'cpu':self._xx2k_one2one_cpu}
        return func.get(self.processor)(*args, **kwargs)
    def _k2xx_one2one_cpu(self, k):
        """
        Private: the inverse FFT and image cropping
                 (which is the reverse of _xx2k() method)
        """
#         dd = numpy.size(self.Kd)

        k = numpy.fft.ifftn(k, axes=self.ft_axes)
        xx = numpy.zeros(self.st['Nd'], dtype=self.dtype, order='C')
        # for bat in range(0, self.batch):
        xx.ravel()[self.NdCPUorder] = k.ravel()[self.KdCPUorder]
        # xx = xx[crop_slice_ind(self.Nd)]
        return xx

    def _xx2x_cpu(self, xx):
        """
        Private: rescaling, which is identical to the  _x2xx() method
        """
        x = self._x2xx_cpu(xx)
        return x
    def k2y2k(self, *args, **kwargs):
        func = {'cpu': self._k2y2k_cpu}
        return func.get(self.processor)(*args, **kwargs)
    def _k2y2k_cpu(self, k):
        """
        Private: the integrated interpolation-gridding by the Sparse
                 Matrix-Vector Multiplication
        """

        Xk = self._k2vec_cpu(k)
        # k = self.spHsp.dot(Xk)
        # k = self.spH.dot(self.sp.dot(Xk))
        k = self._y2vec_cpu(self._vec2y_cpu(Xk))
        k = self._vec2k_cpu(k)
        return k
    
####################################
#     HSA code
####################################

    @push_cuda_context
    def _reset_sense_hsa(self):
        self.volume['gpu_coil_profile'].fill(1.0)

    @push_cuda_context
    def _set_sense_hsa(self, coil_profile):
        if coil_profile.shape != self.multi_Nd:
            print('The shape of coil_profile is ', coil_profile.shape)
            print('But it should be', self.Nd + (self.batch, ))
            raise ValueError
        else:
            self.volume['gpu_coil_profile'] = self.thr.to_device(
                coil_profile.astype(self.dtype))
            if self.verbosity > 0:
                print('Successfully loading coil sensitivities!')

        # if coil_profile.shape == self.Nd + (self.batch, ):

    @push_cuda_context
    def to_device(self, image, shape=None):

        g_image = self.thr.array(image.shape, dtype=self.dtype)
        self.thr.to_device(image.astype(self.dtype), dest=g_image)
        return g_image    
    
    @push_cuda_context
    def _s2x_hsa(self, s):
        x = self.thr.array(self.multi_Nd, dtype=self.dtype)

        self.prg.cPopulate(
            self.batch,
            self.Ndprod,
            s,
            x,
            local_size=None,
            global_size=int(self.batch * self.Ndprod))

        self.prg.cMultiplyVecInplace(
            numpy.uint32(1),
            self.volume['gpu_coil_profile'],
            x,
            local_size=None,
            global_size=int(self.batch*self.Ndprod))

        return x    
    
    @push_cuda_context
    def _x2xx_hsa(self, x):

        xx = self.thr.array(x.shape, dtype=self.dtype)
        self.thr.copy_array(x, dest=xx, )

        self.prg.cTensorMultiply(numpy.uint32(self.batch),
                                 numpy.uint32(self.tSN['Tdims']),
                                 self.tSN['Td'],
                                 self.tSN['Td_elements'],
                                 self.tSN['invTd_elements'],
                                 self.tSN['tensor_sn'],
                                 xx,
                                 numpy.uint32(0),
                                 local_size=None,
                                 global_size=int(self.batch*self.Ndprod))
        # self.thr.synchronize()
        return xx    
    
    @push_cuda_context
    def _xx2k_hsa(self, xx):

        """
        Private: oversampled FFT on the heterogeneous device

        First, zeroing the self.k_Kd array
        Second, copy self.x_Nd array to self.k_Kd array by cSelect
        Third, inplace FFT
        """
        k = self.thr.array(self.multi_Kd, dtype=self.dtype)
        
        k.fill(0)

        self.prg.cTensorCopy(
             self.batch,
             numpy.uint32(self.ndims),
             self.volume['Nd_elements'],
             self.volume['Kd_elements'],
             self.volume['invNd_elements'],
             xx,
             k,
             numpy.int32(1),  # Directions: Nd -> Kd, 1; Kd -> Nd, -1
             local_size=None,
             global_size=int(self.Ndprod))
        self.fft(k, k, inverse=False)
#         self.thr.synchronize()
        return k    
    
    @push_cuda_context
    def _k2y_hsa(self, k):
        """
        Private: interpolation by the Sparse Matrix-Vector Multiplication
        """
        # if self.parallel_flag is 1:
        #     y =self.thr.array((self.st['M'], self.batch),
        #                       dtype=self.dtype).fill(0)
        # else:
        #     y =self.thr.array( (self.st['M'], ), dtype=self.dtype).fill(0)
        y = self.thr.array(self.multi_M, dtype=self.dtype).fill(0)
        self.prg.pELL_spmv_mCoil(
                            self.batch,
                            self.pELL['nRow'],
                            self.pELL['prodJd'],
                            self.pELL['sumJd'],
                            self.pELL['dim'],
                            self.pELL['Jd'],
                            # self.pELL_currsumJd,
                            self.pELL['meshindex'],
                            self.pELL['kindx'],
                            self.pELL['udata'],
                            k,
                            y,
                            local_size=int(self.wavefront),
                            global_size=int(self.pELL['nRow'] *
                                            self.batch * self.wavefront)
                            )
#         self.thr.synchronize()
        return y    
    
    @push_cuda_context
    def _y2k_hsa(self, y):
        """
        Private: gridding by the Sparse Matrix-Vector Multiplication
        However, serial atomic add is far too slow and inaccurate.
        """

#         kx = self.thr.array(self.multi_Kd, dtype=numpy.float32).fill(0.0)
#         ky = self.thr.array(self.multi_Kd, dtype=numpy.float32).fill(0.0)
        k = self.thr.array(self.multi_Kd, dtype=numpy.complex64).fill(0.0)
        res = self.thr.array(self.multi_Kd, dtype=numpy.complex64).fill(0.0)
        # array which saves the residue of two sum
        
        self.prg.pELL_spmvh_mCoil(
                            self.batch,
                            self.pELL['nRow'],
                            self.pELL['prodJd'],
                            self.pELL['sumJd'],
                            self.pELL['dim'],
                            self.pELL['Jd'],
                            self.pELL['meshindex'],
                            self.pELL['kindx'],
                            self.pELL['udata'],
#                             kx, ky,
                            k,
                            res,
                            y,
                            local_size=None,
                            global_size=int(self.pELL['nRow']*self.batch )#*
#                                             int(self.pELL['prodJd']) * int(self.batch))
                            )

        return k + res    
    
    
    @push_cuda_context
    def _k2xx_hsa(self, k):
        """
        Private: the inverse FFT and image cropping (which is the reverse of
        _xx2k() method)
        """

        self.fft(k, k, inverse=True)

        xx = self.thr.array(self.multi_Nd, dtype=self.dtype)
        xx.fill(0)

        self.prg.cTensorCopy(
                             self.batch,
                             numpy.uint32(self.ndims),
                             self.volume['Nd_elements'],
                             self.volume['Kd_elements'],
                             self.volume['invNd_elements'],
                             k,
                             xx,
                             numpy.int32(-1),
                             local_size=None,
                             global_size=int(self.Ndprod))
        return xx    
    
    
    @push_cuda_context
    def _xx2x_hsa(self, xx):
        x = self.thr.array(xx.shape, dtype=self.dtype)
        self.thr.copy_array(xx, dest=x, )

        self.prg.cTensorMultiply(numpy.uint32(self.batch),
                                 numpy.uint32(self.tSN['Tdims']),
                                 self.tSN['Td'],
                                 self.tSN['Td_elements'],
                                 self.tSN['invTd_elements'],
                                 self.tSN['tensor_sn'],
                                 x,
                                 numpy.uint32(0),
                                 local_size=None,
                                 global_size=int(self.batch *
                                                 self.Ndprod))
        
        return x    
    
    @push_cuda_context
    def _x2s_hsa(self, x):
        s = self.thr.array(self.st['Nd'], dtype=self.dtype)

        self.prg.cMultiplyConjVecInplace(
            numpy.uint32(1),
            self.volume['gpu_coil_profile'],
            x,
            local_size=None,
            global_size=int(self.batch*self.Ndprod))

        self.prg.cAggregate(
            self.batch,
            self.Ndprod,
            x,
            s,
            local_size=int(self.wavefront),
            global_size=int(self.batch*self.Ndprod*self.wavefront))
 
        return s    
    
    @push_cuda_context
    def _selfadjoint_one2many2one_hsa(self, gx):
        """
        selfadjoint_one2many2one NUFFT (Teplitz) on the heterogeneous device

        :param gx: The input gpu array, with size=Nd
        :type gx: reikna gpu array with dtype =numpy.complex64
        :return: gx: The output gpu array, with size=Nd
        :rtype: reikna gpu array with dtype =numpy.complex64
        """

        gy = self._forward_one2many_hsa(gx)
        gx2 = self._adjoint_many2one_hsa(gy)
        del gy
        return gx2    
    
    @push_cuda_context
    def _selfadjoint_hsa(self, gx):
        """
        selfadjoint NUFFT (Toeplitz) on the heterogeneous device

        :param gx: The input gpu array, with size=Nd
        :type gx: reikna gpu array with dtype =numpy.complex64
        :return: gx: The output gpu array, with size=Nd
        :rtype: reikna gpu array with dtype =numpy.complex64
        """

        gy = self._forward_hsa(gx)
        gx2 = self._adjoint_hsa(gy)
        del gy
        return gx2    
    
    @push_cuda_context
    def _forward_hsa(self, gx):
        """
        Forward NUFFT on the heterogeneous device

        :param gx: The input gpu array, with size = Nd
        :type gx: reikna gpu array with dtype = numpy.complex64
        :return: gy: The output gpu array, with size = (M,)
        :rtype: reikna gpu array with dtype = numpy.complex64
        """
        try:
            xx = self._x2xx_hsa(gx)
        except:  # gx is not a gpu array
            try:
                warnings.warn('The input array may not be a GPUarray '
                              'Automatically moving the input array to gpu, '
                              'which is throttled by PCIe.', UserWarning)
                px = self.to_device(gx, )
                # pz = self.thr.to_device(numpy.asarray(gz.astype(self.dtype),
                #                                       order = 'C' ))
                xx = self._x2xx_hsa(px)
            except:
                if gx.shape != self.Nd + (self.batch, ):
                    warnings.warn('Shape of the input is ' + str(gx.shape) +
                                  ' while it should be ' +
                                  str(self.Nd+(self.batch, )), UserWarning)
                raise

        k = self._xx2k_hsa(xx)
        del xx
        gy = self._k2y_hsa(k)
        del k
        return gy    
    
    @push_cuda_context
    def _forward_one2many_hsa(self, s):
        try:
            x = self._s2x_hsa(s)
        except:  # gx is not a gpu array
            try:
                warnings.warn('In s2x(): The input array may not be '
                              'a GPUarray. Automatically moving the input'
                              ' array to gpu, which is throttled by PCIe.',
                              UserWarning)
                ps = self.to_device(s, )
                # px = self.thr.to_device(numpy.asarray(x.astype(self.dtype),
                #                                       order = 'C' ))
                x = self._s2x_hsa(ps)
            except:
                if s.shape != self.Nd:
                    warnings.warn('Shape of the input is ' + str(x.shape) +
                                  ' while it should be ' +
                                  str(self.Nd), UserWarning)
                raise

        y = self._forward_hsa(x)
        return y    
    
    @push_cuda_context
    def _adjoint_many2one_hsa(self, y):
        try:
            x = self._adjoint_hsa(y)
        except:  # gx is not a gpu array
            try:
                if self.verbosity > 0:
                    print('y.shape = ', y.shape)
                warnings.warn('In adjoint_many2one(): The input array may not '
                              'be a GPUarray. Automatically moving the input'
                              ' array to gpu, which is throttled by PCIe.',
                              UserWarning)
                py = self.to_device(y, )
                # py = self.thr.to_device(numpy.asarray(y.astype(self.dtype),
                #                                       order = 'C' ))
                x = self._adjoint_hsa(py)
            except:
                print('Failed at self.adjoint_many2one! Please check the gy'
                      ' shape, type and stride.')
                raise
        # z = self.adjoint(y)
        s = self._x2s_hsa(x)
        return s    
    
    @push_cuda_context
    def _adjoint_hsa(self, gy):
        """
        Adjoint NUFFT on the heterogeneous device

        :param gy: The input gpu array, with size=(M,)
        :type: reikna gpu array with dtype =numpy.complex64
        :return: gx: The output gpu array, with size=Nd
        :rtype: reikna gpu array with dtype =numpy.complex64
        """
        try:
            k = self._y2k_hsa(gy)
        except:  # gx is not a gpu array
            try:
                warnings.warn('In adjoint(): The input array may not '
                              'be a GPUarray. Automatically moving the input'
                              ' array to gpu, which is throttled by PCIe.',
                              UserWarning)
                py = self.to_device(gy, )
                # py = self.thr.to_device(numpy.asarray(gy.astype(self.dtype),
                #                         order = 'C' ))
                k = self._y2k_hsa(py)
            except:
                print('Failed at self.adjont! Please check the gy shape, '
                      'type, stride.')
                raise

#             k = self.y2k(gy)
        xx = self._k2xx_hsa(k)
        del k
        gx = self._xx2x_hsa(xx)
        del xx
        return gx   
    
    @push_cuda_context
    def release(self):
        del self.volume
        del self.prg
        del self.pELL
        self.thr.release()
        del self.thr     
        
    @push_cuda_context
    def _solve_hsa(self, gy, solver=None, *args, **kwargs):
        """
        The solver of NUFFT_hsa

        :param gy: data, reikna array, (M,) size
        :param solver: could be 'cg', 'L1TVOLS', 'L1TVLAD'
        :param maxiter: the number of iterations
        :type gy: reikna array, dtype = numpy.complex64
        :type solver: string
        :type maxiter: int
        :return: reikna array with size Nd
        """
        from ..linalg.solve_hsa import solve

        try:
            return solve(self,  gy,  solver, *args, **kwargs)
        except:
            try:
                warnings.warn('In solve(): The input array may not '
                              'be a GPUarray. Automatically moving the input'
                              ' array to gpu, which is throttled by PCIe.',
                              UserWarning)
                py = self.to_device(gy, )
                return solve(self,  py,  solver, *args, **kwargs)
            except:
                if numpy.ndarray == type(gy):
                    print("Input gy must be a reikna array with dtype ="
                          " numpy.complex64")
                    raise  # TypeError
                else:
                    print("wrong")
                    raise  # TypeError        