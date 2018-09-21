"""
cSparseMatVec
==============================================
KERNEL void cCSR_spmv(    
      const    uint    numRow,
      GLOBAL_MEM const uint *rowDelimiters, 
      GLOBAL_MEM const uint *cols,
      GLOBAL_MEM const float2 *val,
      GLOBAL_MEM const float2 *vec, 
      GLOBAL_MEM float2 *out)
      
Offload Sparse Matrix Vector Multiplication to heterogeneous devices.
Note: In CUDA, += operator can cause problems. Here we use explicit add operator.
"""

R="""
KERNEL void cCSR_spmv(    
      const    uint    numRow,
      GLOBAL_MEM const uint *rowDelimiters, 
      GLOBAL_MEM const uint *cols,
      GLOBAL_MEM const float2 *val,
      GLOBAL_MEM const float2 *vec, 
      GLOBAL_MEM float2 *out)
{   
    const uint t = get_local_id(0);
    const uint vecWidth=${LL};
    // Thread ID within wavefront
    const uint id = t & (vecWidth-1);
    // One row per wavefront
    uint vecsPerBlock=get_local_size(0)/vecWidth;
    uint myRow=(get_group_id(0)*vecsPerBlock) + (t/ vecWidth);
    LOCAL_MEM float2 partialSums[${LL}];
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    
    partialSums[t] = zero;
    float2  y= zero;
    if (myRow < numRow)
    {
     const uint vecStart = rowDelimiters[myRow];
     const uint vecEnd = rowDelimiters[myRow+1];            
     for (uint j = vecStart+id;  j<vecEnd; j += vecWidth)
     {
          const uint col = cols[j];
          const float2 spdata=val[j];
          const float2 vecdata=vec[col];                        
          y.x=spdata.x*vecdata.x - spdata.y*vecdata.y;
          y.y=spdata.y*vecdata.x + spdata.x*vecdata.y;
          partialSums[t] = partialSums[t] + y;
      }

      LOCAL_BARRIER; 
      //__syncthreads();
      //barrier(CLK_LOCAL_MEM_FENCE);
      // Reduce partial sums
      uint bar = vecWidth / 2;
      while(bar > 0)
      {
           if (id < bar)
          partialSums[t] = partialSums[t] + partialSums[t+bar];
           
           //barrier(CLK_LOCAL_MEM_FENCE);
           //__syncthreads();
           LOCAL_BARRIER;
           bar = bar / 2;
      }            
      // Write result 
      if (id == 0)
      {
       out[myRow]=partialSums[t]; 
      }
     }
    };    
    
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

                                        u.x =  spdata.x*y.x + spdata.y*y.y;
                                       u.y =  - spdata.y*y.x + spdata.x*y.y;
                                       
                                       atomic_add_float(kx+col, u.x);
                                       atomic_add_float(ky+col, u.y);
                            };
                    };
                };    
                
                

KERNEL void cELL_spmvh_scalar(    
      const    uint    nRow,
      const    uint    colWidth, 
      GLOBAL_MEM const uint *cols,
      GLOBAL_MEM const float2 *data,
      GLOBAL_MEM float *kx,
      GLOBAL_MEM float *ky, 
      GLOBAL_MEM const float2 *out)
{      uint myRow= get_global_id(0);
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    if (myRow < nRow)
    {      
     float2  u= zero;
     // out[myRow] =zero; 
     const float2 vecdata=out[myRow];
     for (uint j = myRow *  colWidth;  j< (myRow + 1) *  colWidth; j ++)
     {
          uint col = cols[j];
          float2 spdata=data[j];
                   
          u.x =spdata.x*vecdata.x + spdata.y*vecdata.y;
          u.y =-spdata.y*vecdata.x + spdata.x*vecdata.y;
          
          // vec[col] += u;
          atomic_add_float(kx+col, u.x);
          atomic_add_float(ky+col, u.y);
     }
     //LOCAL_BARRIER;
        //out[myRow]= y;
    }
    };                        
    
"""
from numpy import uint32
scalar_arg_dtypes=[uint32, None, None, None, None, None]        
