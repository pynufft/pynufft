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
                
                inline void atomic_add_float( 
                GLOBAL_MEM float *ptr, 
                const float temp) 
                {
                // The work-around of AtomicAdd for float
                // lockless add *source += operand 
                // Caution!!!!!!! Use with care! You have been warned!
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
                
              

        
            KERNEL void cCSR_spmvh(    const    uint    n_row,
                                    GLOBAL_MEM const uint *indptr, 
                                    GLOBAL_MEM  const uint *indices,
                                    GLOBAL_MEM  const float2 *data,
                                    // GLOBAL_MEM  float2 *Xx,
                                    GLOBAL_MEM float *kx, 
                                     GLOBAL_MEM float *ky,
                                    GLOBAL_MEM   const float2 *Yx)
                {   

                        uint myRow = get_global_id(0);
                       
                            float2 zero;
                            zero.x=0.0;
                            zero.y=0.0;
                            
                            float2  u = zero;
                            
                     if (myRow < n_row)
                       {   
                            uint vecStart = indptr[myRow];
                            uint vecEnd = indptr[myRow+1];
                           float2 y =  Yx[myRow];
                            for (uint j= vecStart + 0; j < vecEnd;  j+= 1) //vecWidth)
                            {
                                       const uint col = indices[j];
                                        
                                       const float2 spdata =  data[j]; // row

                                        u.x =  spdata.x*y.x - (-spdata.y)*y.y;
                                       u.y =  (- spdata.y)*y.x + spdata.x*y.y;
                                       
                                       atomic_add_float(kx+col, u.x);
                                       atomic_add_float(ky+col, u.y);
                            };
                    };
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