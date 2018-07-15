"""
cCSR_spmvh
==============================================
KERNEL void cCSR_spmvh(    
      const    uint    dim,
      GLOBAL_MEM const uint *rowDelimiters, 
      GLOBAL_MEM const uint *cols,
      GLOBAL_MEM const float2 *val,
      GLOBAL_MEM const float2 *vec, 
      GLOBAL_MEM float2 *out)
      
Offload Sparse Matrix Vector Multiplication to heterogeneous devices.
"""

R="""
                
                KERNEL void AtomicAdd(volatile GLOBAL_MEM float2 *source, const float2 operand) {
                // The work-around of AtomicAdd for float2
                // lockless add *source += operand 
                // Caution!!!!!!! Use with care! You have been warned!
                // Author:  Igor Suhorukov 
                // Source: http://suhorukov.blogspot.co.uk/2011/12/opencl-11-atomic-operations-on-floating.html
                    union {
                        unsigned int intVal;
                        float2 floatVal;
                    } newVal;
                    union {
                        unsigned int intVal;
                        float2 floatVal;
                    } prevVal;
                    do {
                        prevVal.floatVal = *source;
                        newVal.floatVal = prevVal.floatVal + operand;
                    } while (atomic_cmpxchg((volatile GLOBAL_MEM unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
                };
                
                inline void atomic_add_float2( 
                GLOBAL_MEM float2 *ptr, 
                const float2 temp) 
                {
                // The work-around of AtomicAdd for float2
                // lockless add *source += operand 
                // Caution!!!!!!! Use with care! You have been warned!
                // Source: https://github.com/clMathLibraries/clSPARSE/blob/master/src/library/kernels/csrmv_adaptive.cl
                    union {
                        unsigned int intVal;
                        float2 floatVal;
                    } newVal;
                    union {
                        unsigned int intVal;
                        float2 floatVal;
                    } prevVal;
                    do {
                        prevVal.floatVal = *ptr;
                        newVal.floatVal = prevVal.floatVal + temp;
                    } while (atomic_cmpxchg((volatile GLOBAL_MEM unsigned int *)ptr, prevVal.intVal, newVal.intVal) != prevVal.intVal);
                };                
                
                inline void atomic_add_float_extended( 
                GLOBAL_MEM float2 *ptr, 
                const float2 temp,
                float2 *old_sum,
                float2 *sumk_s) 
                {
                // The work-around of AtomicAdd for float2
                // lockless add *source += operand 
                // Caution!!!!!!! Use with care! You have been warned!
                // Source: https://github.com/clMathLibraries/clSPARSE/blob/master/src/library/kernels/csrmv_adaptive.cl
                    union {
                        unsigned int intVal;
                        float2 floatVal;
                    } newVal;
                    union {
                        unsigned int intVal;
                        float2 floatVal;
                    } prevVal;
                    do {
                        prevVal.floatVal = *ptr;
                        newVal.floatVal = prevVal.floatVal + temp;
                    } while (atomic_cmpxchg((volatile GLOBAL_MEM unsigned int *)ptr, prevVal.intVal, newVal.intVal) != prevVal.intVal);
                    
                    *old_sum = prevVal.floatVal;
                    *sumk_s = newVal.floatVal;
                };
                
                inline void atomic_two_sum_float(GLOBAL_MEM float2 *x_ptr, 
                                                                                float2 b, 
                                                                               float2 *r)
                {
                float2 sumk_s = 0.;
                float2 a;
                
                atomic_add_float_extended(x_ptr, b, &a, &sumk_s);
                float2 a_prime = sumk_s - b;
                float2 b_prime = sumk_s - a_prime;
                float2 delta_a = a - a_prime;
                float2 delta_b = b-b_prime;
                
                *r = delta_a + delta_b;
                };
        
                      KERNEL void cCSR_spmvh(    const    uint    n_row,
                                    GLOBAL_MEM const uint *indptr, 
                                    GLOBAL_MEM  const uint *indices,
                                    GLOBAL_MEM  const float2 *data,
                                    GLOBAL_MEM  float2 *Xx, 
                                    GLOBAL_MEM   const float2 *Yx,
                                    GLOBAL_MEM  float2 *Xx2)
                {   
                        // const uint t = get_local_id(0);
                        // const uint vecWidth=${LL};
                        // Thread ID within wavefront
                        // const uint id = t & (vecWidth-1);
                        // One row per wavefront
                        // uint vecsPerBlock = get_local_size(0) / vecWidth;
                        // uint myRow = (get_group_id(0) * vecsPerBlock) + (t / vecWidth);
                        uint myRow = get_global_id(0);
                        float2 tmp_err;
                       // LOCAL_MEM  float2 partialSums[${LL}];
                        
                            float2 zero;
                            zero.x=0.0;
                            zero.y=0.0;
                            
                            float2  y=zero;
                            
                    if (myRow < n_row)
                        {
                            uint vecStart = indptr[myRow];
                            uint vecEnd = indptr[myRow+1];
                            
                           float2 vecdata =  Yx[myRow];

                            
                            for (uint j= vecStart + 0; j < vecEnd;  j+= 1) //vecWidth)
                            {
                                        uint col = indices[j];
                                        
                                         //float2 tmp_X;
                                        // tmp_X = Xx[col];
                                        float2 spdata =  data[j]; // row

                                        y.x =  spdata.x*vecdata.x +  spdata.y*vecdata.y;
                                       y.y =  - spdata.y*vecdata.x + spdata.x*vecdata.y;
                                       
                                       // Xx[col] += y;
                                        // replaced by atomic add (reimplemented with two-sum)
                                        // ATOMIC ADD: lockless parallel atomic_add using CAP (comapre and swap)
                                                                               
                                        //AtomicAdd(Xx+col, y);
                                        atomic_two_sum_float(Xx+col, y, &tmp_err);
                                        atomic_add_float2(Xx2+col, tmp_err);
                                        
                            };
                          // __syncthreads();
                         //LOCAL_BARRIER;           
                    };
                     //barrier(CLK_GLOBAL_MEM_FENCE);      
                };
                
                KERNEL void zeroing(GLOBAL_MEM float2* x)
                {uint gid = get_global_id(0);
                float2 z;
                z.x = 0.0;
                z.y = 0.0;
                x[gid] = z;
                }; 
                KERNEL void AddVec( 
                GLOBAL_MEM float2 *a,
                GLOBAL_MEM float2 *b)
                {const int i = get_global_id(0);
                a[i] += b[i];
                //barrier(CLK_GLOBAL_MEM_FENCE);
                };
                """
from numpy import uint32
scalar_arg_dtypes=[uint32, None, None, None, None, None]        
# S="""
# 
# //  summation-error corrected csr_spmv_scalar_kernel 
# // Modified from cuSPARSE and the csrspmv_general.cl in clSPARSE package
# // Floating point errors of repeated summation have been corrected by the 6FLOPS algorithm
# KERNEL  void cSparseMatVec(      const       uint num_rows,
#                                              GLOBAL_MEM const uint *ptr, 
#                                             GLOBAL_MEM  const uint *indices,
#                                             GLOBAL_MEM const float2 *data,
#                                             GLOBAL_MEM const float2 *x, 
#                                            GLOBAL_MEM float2 *y)
# {  //LOCAL_MEM float2  *vals;
# const uint i = get_global_id(0);
#     if ( i < num_rows ){
#       float2 dot ;
#       dot.x=0.0;
#       dot.y=0.0;
#            int row_start = ptr[ i ];
#            int row_end = ptr[ i +1];
#            
#         float2 sumk_err;
#               sumk_err.x=0.0;
#               sumk_err.y=0.0;
#         float2 y2;
#             y2.x=0.0;
#             y2.y=0.0;
#         float2 sumk_s;
#         float2 bp;
#         
#            for ( int jj = row_start ; jj < row_end ; jj ++)
#                    {
#                    uint idx = indices[jj];
#                   // dot += ${mul}(data[ jj ] , x[ idx]);
#              //y2 =${mul}(data[ jj ] , x[ idx]);
#              y2.x = data[ jj ].x* x[ idx].x -  data[ jj ].y* x[ idx].y;
#              y2.y = data[ jj ].x* x[ idx].y+  data[ jj ].y* x[ idx].x;
#              sumk_s = dot+y2;
#              bp = sumk_s - dot;
#              sumk_err = sumk_err + ((dot - (sumk_s - bp)) + (y2 - bp));
#              dot = sumk_s;                   
#                    }
#                float2 new_error ;
#                new_error.x=0.0;
#                new_error.y=0.0;
#         
#             y2 =sumk_err;
#             sumk_s = dot+y2;
#             bp = sumk_s -dot;
#             new_error = new_error + ((dot - (sumk_s - bp)) + (y2 - bp));
#             dot = sumk_s;
#            y[ i ] = dot ;
#     };
# //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_FENCE);
# };
# 
# """      