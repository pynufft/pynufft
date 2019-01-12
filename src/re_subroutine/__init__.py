"""
Metaprogramming subroutines (using reikna, pyopencl, pycuda)
========================================================
"""

from __future__ import absolute_import # Python2 compatibility
from . import cFill, cMultiplyScalar, cCopy, cTensorMultiply, cTensorCopy, cAddScalar, cAddVec,  cSelect, cMultiplyVec, cMultiplyConjVecInplace, cMultiplyVecInplace, cMultiplyRealInplace, cMultiplyConjVec, cDiff, cSqrt, cAnisoShrink, cHypot, cSpmv, cSpmvh, atomic_add, cHadamard

def create_kernel_sets(API):
    kernel_sets = ( cFill.R + cMultiplyScalar.R + 
                        cCopy.R + cTensorCopy.R + cHypot.R +cTensorMultiply.R + 
                        cAddScalar.R + 
                        cSelect.R + 
                        cMultiplyConjVec.R + 
                        cAddVec.R+  
                        cMultiplyVecInplace.R + cMultiplyConjVecInplace.R +cMultiplyRealInplace.R + 
                        cDiff.R+ cSqrt.R+ cAnisoShrink.R+ cMultiplyVec.R + cSpmv.R + cSpmvh.R + cHadamard.R)
    if 'cuda' is API:
        print('Select cuda interface')
        kernel_sets =  atomic_add.cuda_add + kernel_sets
    elif 'ocl' is API:
        print("Selecting opencl interface")
        kernel_sets =  atomic_add.ocl_add + kernel_sets
    return kernel_sets

"""

- KERNEL void cAbsVec( GLOBAL_MEM const float2 *indata,  GLOBAL_MEM  float2 *outdata)

    Abs of indata.

    :input float2 *indata: 
    :return float2  *outdata: 

- KERNEL void cAddScalar(const float2 CA,  GLOBAL_MEM float2 *CX)

    Offload add to heterogeneous devices.

    :input float2 CA: 
    :return float2 *CX:

- KERNEL void cAddVec(   GLOBAL_MEM float2 *a,  GLOBAL_MEM float2 *b, GLOBAL_MEM float2 *dest)    
        
    *dest = *a + *b
        
    :input  float2 *a:
    :input float2 *b:
    :return float2 *dest:        
    
- KERNEL void cAnisoShrink(const  float2 threshold, GLOBAL_MEM const float2 *indata, GLOBAL_MEM  float2 *outdata)
        
    Soft-thresholding of real and imaginery parts. The real part of threshold is used.

    :input float2 threshold: threshold.x is the threshold; threshold.y is zero.
    :input float2 *indata: input data
    :return float2 *outdata:      
    
- KERNEL void cCopy( GLOBAL_MEM  const float2 *CX, GLOBAL_MEM float2 *CY)

    Copy array *CX to array *CY.
    
    :input float2 *CX: input array
    :return float2 *CY: output array

- KERNEL void cDiff(  GLOBAL_MEM const int *order2, GLOBAL_MEM const   float2 *indata, GLOBAL_MEM  float2 *outdata)

     Compute indata[order2[gid]] - indata[gid] (finite difference)
     
     :input int *order2: index of shifted pixel
     :input float2 *indata: input image
     :return float2 *outdata: return 

- KERNEL void cMultiplyConjVec(GLOBAL_MEM float2 *a, GLOBAL_MEM float2 *b, GLOBAL_MEM float2 *dest)

    dest = conj(a) x b
    
    :input float2 *a:
    :input float2 *b:
    :return float2 *dest: 
    
- KERNEL void cMultiplyScalar(const float2 CA, GLOBAL_MEM float2 *CX)

    *CX *= CA 
    
    :input float2 CA:
    :input float2 *CX:
    :return float2 *CX: 

- KERNEL void cMultiplyVec( GLOBAL_MEM float2 *a, GLOBAL_MEM float2 *b, GLOBAL_MEM float2 *dest)

    Array multiplication (*dest = *a x *b)
    
    :input float2 *a: array1
    :input float2 *b: array2
    :return float2 *dest: output array
    
- KERNEL void cMultiplyVecInplace(GLOBAL_MEM float2 *a, GLOBAL_MEM float2 *outb)

    Inplace multiplciation (*outb *= *a)
    
    :input float2 *a: array1
    :input float2 *outb: array2
    :return float2 *outb: array2
    
- KERNEL void cSelect( GLOBAL_MEM const  int *order1, GLOBAL_MEM const  int *order2, GLOBAL_MEM const float2 *indata, GLOBAL_MEM       float2 *outdata)

    Copy indata[order1] to outdata[order2]. outdata[order2[gid]] = indata[order1[gid]]
    
    :input int *order1: index of indata
    :input int *order2: index of outdata
    :input float2 *indata:
    :return float2 *outdata:
      
- KERNEL void cCSR_spmv(    
      const    uint    dim,
      GLOBAL_MEM const uint *rowDelimiters, 
      GLOBAL_MEM const uint *cols,
      GLOBAL_MEM const float2 *val,
      GLOBAL_MEM const float2 *vec, 
      GLOBAL_MEM float2 *out)
      
      CSR matrix Sparse Matrix Vector Multiplication on heterogeneous devices.
      
      :input uint dim: number of rows
      :input uint *rowDelimiters:
      :input uint *cols:
      :input float2 *val:
      :input float2 *vec:
      :return float2 *out:      
      
- KERNEL void cSqrt( GLOBAL_MEM  float2 *CX)

    Inplace Sqrt
    
    :input float2 *CX:
    :return float2 *CX:
"""