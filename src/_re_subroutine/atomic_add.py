
ocl_add = """
// #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
KERNEL void atomic_add_float( 
        GLOBAL_MEM float *ptr, 
        const float temp) 
{
// The work-around of AtomicAdd for float
// lockless add *source += operand 
// Caution!!!!!!! Use with care! You have been warned!
// http://simpleopencl.blogspot.com/2013/05/atomic-operations-and-floats-in-opencl.html
// Source: https://github.com/clMathLibraries/clSPARSE/blob/master/src/library/kernels/csrmv_adaptive.cl
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    
    do {
        prevVal.floatVal = *ptr;
        newVal.floatVal = prevVal.floatVal + temp;
    } while (atomic_cmpxchg((volatile GLOBAL_MEM unsigned int *)ptr, prevVal.intVal, newVal.intVal) != prevVal.intVal);
};         
"""

cuda_add = """

__device__ void atomic_add_float( 
        GLOBAL_MEM float *ptr, 
        const float temp) 
{ // Wrapper around CUDA atomicAdd();
atomicAdd(ptr, temp); 
};     

"""