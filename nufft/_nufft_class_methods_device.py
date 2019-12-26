####################################
#     HSA code
####################################

import numpy
# import reikna
from functools import wraps as _wraps
from ..src._helper import helper, helper1

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


def _init__device(self, device_indx=None):
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
    >>> NufftObj = NUFFT_device(API='cuda', platform_number=0,
                                     device_number=0, verbosity=0)
    """

    self.dtype = numpy.complex64
    self.verbosity = 0#verbosity

    from reikna import cluda
    import reikna.transformations
    from reikna.cluda import functions, dtypes
#         try:  # try to create api/platform/device using the given parameters
    API = device_indx[0]
    self.API = API
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
    kernel_sets = re_subroutine.create_kernel_sets(self.API)

    prg = self.thr.compile(kernel_sets,
                           render_kwds=dict(LL=str(self.wavefront)),
                           fast_math=False)
    self.prg = prg
    self.processor = 'hsa'



def _plan_device(self, om, Nd, Kd, Jd, ft_axes=None, batch=None, radix=None):
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
    >>> NufftObj = NUFFT_device()
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
    self._offload_device()

    return 0    

@push_cuda_context
def _offload_device(self):  # API, platform_number=0, device_number=0):
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

@push_cuda_context
def _set_wavefront_device(self, wf):
#         try:
    self.wavefront = int(wf)#api.DeviceParameters(device).warp_size
    if self.verbosity > 0:
        print('Wavefront of OpenCL (as wrap of CUDA) = ', self.wavefront)

    from ..src import re_subroutine  # import create_kernel_sets
    kernel_sets = re_subroutine.create_kernel_sets(self.API)

    prg = self.thr.compile(kernel_sets,
                           render_kwds=dict(LL=str(self.wavefront)),
                           fast_math=False)
    self.prg = prg

@push_cuda_context
def _reset_sense_device(self):
    self.volume['gpu_coil_profile'].fill(1.0)

@push_cuda_context
def _set_sense_device(self, coil_profile):
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
def to_device(self, x, shape=None):
    gx = self.thr.to_device(x.copy().astype(self.dtype))
#         g_image = self.thr.array(image.shape, dtype=self.dtype)
#         self.thr.to_device(image.astype(self.dtype), dest=g_image)
    return gx
 
@push_cuda_context
def to_host(self, data):
    return data.get() 

@push_cuda_context
def _s2x_device(self, s):
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
def _x2xx_device(self, x):

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
def _xx2k_device(self, xx):

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
def _k2y_device(self, k):
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
def _y2k_device(self, y):
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
def _k2xx_device(self, k):
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
def _xx2x_device(self, xx):
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
def _x2s_device(self, x):
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
def _selfadjoint_one2many2one_device(self, gx):
    """
    selfadjoint_one2many2one NUFFT (Teplitz) on the heterogeneous device

    :param gx: The input gpu array, with size=Nd
    :type gx: reikna gpu array with dtype =numpy.complex64
    :return: gx: The output gpu array, with size=Nd
    :rtype: reikna gpu array with dtype =numpy.complex64
    """

    gy = self._forward_one2many_device(gx)
    gx2 = self._adjoint_many2one_device(gy)
    del gy
    return gx2    

@push_cuda_context
def _selfadjoint_device(self, gx):
    """
    selfadjoint NUFFT (Toeplitz) on the heterogeneous device

    :param gx: The input gpu array, with size=Nd
    :type gx: reikna gpu array with dtype =numpy.complex64
    :return: gx: The output gpu array, with size=Nd
    :rtype: reikna gpu array with dtype =numpy.complex64
    """

    gy = self._forward_device(gx)
    gx2 = self._adjoint_device(gy)
    del gy
    return gx2    

@push_cuda_context
def _forward_device(self, gx):
    """
    Forward NUFFT on the heterogeneous device

    :param gx: The input gpu array, with size = Nd
    :type gx: reikna gpu array with dtype = numpy.complex64
    :return: gy: The output gpu array, with size = (M,)
    :rtype: reikna gpu array with dtype = numpy.complex64
    """
    xx = self._x2xx_device(gx)
#         try:
#             xx = self._x2xx_device(gx)
#         except:  # gx is not a gpu array
#             try:
#                 warnings.warn('The input array may not be a GPUarray '
#                               'Automatically moving the input array to gpu, '
#                               'which is throttled by PCIe.', UserWarning)
#                 px = self.to_device(gx, )
#                 # pz = self.thr.to_device(numpy.asarray(gz.astype(self.dtype),
#                 #                                       order = 'C' ))
#                 xx = self._x2xx_device(px)
#             except:
#                 if gx.shape != self.Nd + (self.batch, ):
#                     warnings.warn('Shape of the input is ' + str(gx.shape) +
#                                   ' while it should be ' +
#                                   str(self.Nd+(self.batch, )), UserWarning)
#                 raise

    k = self._xx2k_device(xx)
    del xx
    gy = self._k2y_device(k)
    del k
    return gy    

@push_cuda_context
def _forward_one2many_device(self, s):
    x = self._s2x_device(s)
    y = self._forward_device(x)
    return y    

@push_cuda_context
def _adjoint_many2one_device(self, y):
    x = self._adjoint_device(y)
    s = self._x2s_device(x)
    return s    

@push_cuda_context
def _adjoint_device(self, gy):
    """
    Adjoint NUFFT on the heterogeneous device

    :param gy: The input gpu array, with size=(M,)
    :type: reikna gpu array with dtype =numpy.complex64
    :return: gx: The output gpu array, with size=Nd
    :rtype: reikna gpu array with dtype =numpy.complex64
    """
    k = self._y2k_device(gy)

    xx = self._k2xx_device(k)
    del k
    gx = self._xx2x_device(xx)
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
def _solve_device(self, gy, solver=None, *args, **kwargs):
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
    from ..linalg.solve_device import solve
    return solve(self,  gy,  solver, *args, **kwargs)