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
    A super class of cpu and gpu NUFFT functions. 
    Note: NUFFT does NOT inherit NUFFT_cpu and NUFFT_hsa.
    Multi-coil or single-coil memory reduced NUFFT.

    """
    #import cpu codes
    from ._nufft_class_methods_cpu import _init__cpu, _plan_cpu, _precompute_sp_cpu, _reset_sense_cpu, _set_sense_cpu, _forward_one2many_cpu, _adjoint_many2one_cpu, _solve_cpu, _forward_cpu, _adjoint_cpu, _selfadjoint_one2many2one_cpu, _selfadjoint_cpu, _selfadjoint2_cpu, _x2xx_cpu, _xx2k_cpu, _xx2k_one2one_cpu, _k2vec_cpu, _vec2y_cpu, _k2y_cpu, _y2vec_cpu, _vec2k_cpu, _y2k_cpu, _k2xx_cpu, _k2xx_one2one_cpu, _xx2x_cpu, _k2y2k_cpu
    # import host codes
    from ._nufft_class_methods_cpu import  _forward_host,  _adjoint_host, _selfadjoint_host,  _solve_host, _xx2k_host, _k2xx_host, _x2xx_host, _xx2x_host, _k2y_host,  _y2k_host, _adjoint_many2one_host, _forward_one2many_host, _selfadjoint_one2many2one_host, _reset_sense_host, _set_sense_host
    # import device codes
    from ._nufft_class_methods_device import _init__device, _plan_device,  _set_wavefront_device, _offload_device, _reset_sense_device, _set_sense_device, to_device, to_host, _s2x_device, _x2xx_device, _xx2k_device, _k2y_device, _y2k_device, _k2xx_device, _xx2x_device, _x2s_device, _selfadjoint_one2many2one_device,  _selfadjoint_device, _forward_device, _forward_one2many_device, _adjoint_many2one_device, _adjoint_device, release, _solve_device, release
    
    # legacy codes (csr format for device)
    from ._nufft_class_methods_device import _y2k_legacy, _k2y_legacy, _forward_legacy, _adjoint_legacy, _selfadjoint_legacy, _forward_one2many_legacy, _adjoint_many2one_legacy, _selfadjoint_one2many2one_legacy, _plan_legacy, _offload_legacy, _solve_legacy
    from ._nufft_class_methods_cpu import _k2y_legacy_host,  _y2k_legacy_host, _selfadjoint_legacy_host, _forward_legacy_host, _adjoint_legacy_host, _selfadjoint_one2many2one_legacy_host, _forward_one2many_legacy_host, _adjoint_many2one_legacy_host, _solve_legacy_host
    
    def __init__(self, device_indx=None, legacy=None):
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
        func = {'cpu': self._plan_cpu,
                    'hsa': self._plan_device,
                    'hsa_legacy': self._plan_legacy}
        return func.get(self.processor)(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        func = {'cpu': self._forward_cpu, 
                    'hsa': self._forward_host,
                    'hsa_legacy': self._forward_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def adjoint(self, *args, **kwargs):
        func = {'cpu': self._adjoint_cpu,
                'hsa': self._adjoint_host,
                'hsa_legacy': self._adjoint_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def selfadjoint(self, *args, **kwargs):
        func = {'cpu': self._selfadjoint_cpu,
                'hsa': self._selfadjoint_host,
                'hsa_legacy': self._selfadjoint_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def solve(self, *args, **kwargs):
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
    
    def adjoint_many2one(self, *args, **kwargs):
        func = {'cpu': self._adjoint_many2one_cpu,
                'hsa': self._adjoint_many2one_host,
                'hsa_legacy': self._adjoint_many2one_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def forward_one2many(self, *args, **kwargs):
        func = {'cpu': self._forward_one2many_cpu,
                'hsa': self._forward_one2many_host,
                'hsa_legacy': self._forward_one2many_legacy_host}
        return func.get(self.processor)(*args, **kwargs)
    
    def selfadjoint_one2many2one(self, *args, **kwargs):
        func = {'cpu': self._selfadjoint_one2many2one_cpu,
                'hsa': self._selfadjoint_one2many2one_host,
                'hsa_legacy': self._selfadjoint_one2many2one_legacy_host}
        return func.get(self.processor)(*args, **kwargs)   
    
    def k2xx_one2one(self, *args, **kwargs):
        func = {'cpu':self._k2xx_one2one_cpu}
        return func.get(self.processor)(*args, **kwargs)
    
    def xx2k_one2one(self, *args, **kwargs):
        func = {'cpu':self._xx2k_one2one_cpu}
        return func.get(self.processor)(*args, **kwargs) 
    
    def k2y2k(self, *args, **kwargs):
        func = {'cpu': self._k2y2k_cpu}
        return func.get(self.processor)(*args, **kwargs)
    def set_sense(self, *args, **kwargs):
        func = {'cpu': self._set_sense_cpu,
                'hsa': self._set_sense_host,
                'hsa_legacy': self._set_sense_host}
        return func.get(self.processor)(*args, **kwargs)
    def reset_sense(self, *args, **kwargs):
        func = {'cpu': self._reset_sense_cpu,
                'hsa': self._reset_sense_host,
                'hsa_legacy': self._reset_sense_host}
        return func.get(self.processor)(*args, **kwargs)
    