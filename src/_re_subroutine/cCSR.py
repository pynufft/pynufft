"""
cSparseMatVec
==============================================
KERNEL void cCSR_spmv(    
      const    unsigned int    numRow,
      GLOBAL_MEM const unsigned int *rowDelimiters, 
      GLOBAL_MEM const unsigned int *cols,
      GLOBAL_MEM const float2 *val,
      GLOBAL_MEM const float2 *vec, 
      GLOBAL_MEM float2 *out)
      
Offload Sparse Matrix Vector Multiplication to heterogeneous devices.
Note: In CUDA, += operator can cause problems. Here we use explicit add operator.
"""

R="""
KERNEL void cCSR_spmv(    
      const    unsigned int    numRow,
      GLOBAL_MEM const unsigned int *rowDelimiters, 
      GLOBAL_MEM const unsigned int *cols,
      GLOBAL_MEM const float2 *val,
      GLOBAL_MEM const float2 *vec, 
      GLOBAL_MEM float2 *out)
{   
    const unsigned int t = get_local_id(0);
    const unsigned int vecWidth=${LL};
    // Thread ID within wavefront
    const unsigned int id = t & (vecWidth-1);
    // One row per wavefront
    unsigned int vecsPerBlock=get_local_size(0)/vecWidth;
    unsigned int myRow=(get_group_id(0)*vecsPerBlock) + (t/ vecWidth);
    LOCAL_MEM float2 partialSums[${LL}];
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    
    partialSums[t] = zero;
    float2  y= zero;
    if (myRow < numRow)
    {
     const unsigned int vecStart = rowDelimiters[myRow];
     const unsigned int vecEnd = rowDelimiters[myRow+1];            
     for (unsigned int j = vecStart+id;  j<vecEnd; j += vecWidth)
     {
          const unsigned int col = cols[j];
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
      unsigned int bar = vecWidth / 2;
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
                
              

        
            KERNEL void cCSR_spmvh(    const    unsigned int    n_row,
                                    GLOBAL_MEM const unsigned int *indptr, 
                                    GLOBAL_MEM  const unsigned int *indices,
                                    GLOBAL_MEM  const float2 *data,
                                    // GLOBAL_MEM  float2 *Xx,
                                    GLOBAL_MEM float *kx, 
                                     GLOBAL_MEM float *ky,
                                    GLOBAL_MEM   const float2 *Yx)
                {   

                        unsigned int myRow = get_global_id(0);
                       
                            float2 zero;
                            zero.x=0.0;
                            zero.y=0.0;
                            
                            float2  u = zero;
                            
                     if (myRow < n_row)
                       {   
                            unsigned int vecStart = indptr[myRow];
                            unsigned int vecEnd = indptr[myRow+1];
                           float2 y =  Yx[myRow];
                            for (unsigned int j= vecStart + 0; j < vecEnd;  j+= 1) //vecWidth)
                            {
                                       const unsigned int col = indices[j];
                                        
                                       const float2 spdata =  data[j]; // row

                                        u.x =  spdata.x*y.x + spdata.y*y.y;
                                       u.y =  - spdata.y*y.x + spdata.x*y.y;
                                       
                                       atomic_add_float(kx+col, u.x);
                                       atomic_add_float(ky+col, u.y);
                            };
                    };
                };    
                
                

KERNEL void cELL_spmvh_scalar(    
      const    unsigned int    nRow,
      const    unsigned int    colWidth, 
      GLOBAL_MEM const unsigned int *cols,
      GLOBAL_MEM const float2 *data,
      GLOBAL_MEM float *kx,
      GLOBAL_MEM float *ky, 
      GLOBAL_MEM const float2 *out)
{      unsigned int myRow= get_global_id(0);
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    if (myRow < nRow)
    {      
     float2  u= zero;
     // out[myRow] =zero; 
     const float2 vecdata=out[myRow];
     for (unsigned int j = myRow *  colWidth;  j< (myRow + 1) *  colWidth; j ++)
     {
          unsigned int col = cols[j];
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
# from numpy import unsigned int32
# scalar_arg_dtypes=[unsigned int32, None, None, None, None, None]        
