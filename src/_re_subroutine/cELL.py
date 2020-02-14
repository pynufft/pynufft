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

    
KERNEL void cELL_spmv_vector(    
      const    unsigned int    numRow,
      const    unsigned int    colWidth,
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
     const unsigned int vecStart = myRow*colWidth; 
     const unsigned int vecEnd = (myRow + 1)*colWidth;             
     for (unsigned int j = vecStart+id;  j<vecEnd; j += vecWidth)
     {
          const unsigned int col    =    cols[j];
          const float2 spdata    =    val[j];
          const float2 vecdata    =    vec[col];                        
          y.x=spdata.x*vecdata.x - spdata.y*vecdata.y;
          y.y=spdata.y*vecdata.x + spdata.x*vecdata.y;
          partialSums[t] = y + partialSums[t];
      }

      LOCAL_BARRIER; 
      //__syncthreads();
      //barrier(CLK_LOCAL_MEM_FENCE);
      // Reduce partial sums
      unsigned int bar = vecWidth / 2;
      while(bar > 0)
      {
           if (id < bar)
          partialSums[t]= partialSums[t]+partialSums[t+bar];
           
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

KERNEL void cELL_spmv_scalar(    
      const    unsigned int    nRow,
      const    unsigned int    colWidth, 
      GLOBAL_MEM const unsigned int *cols,
      GLOBAL_MEM const float2 *data,
      GLOBAL_MEM const float2 *vec, 
      GLOBAL_MEM float2 *out)
{      unsigned int myRow= get_global_id(0);
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    if (myRow < nRow)
    {      
     float2  y= zero;
     out[myRow] =zero; 
     for (unsigned int j = myRow *  colWidth;  j< (myRow + 1) *  colWidth; j ++)
     //for (unsigned int j = 0;  j<   colWidth; j ++)
     {
          unsigned int col = cols[j];
          float2 spdata=data[j];
          float2 vecdata=vec[col];                        
          y.x +=spdata.x*vecdata.x - spdata.y*vecdata.y;
          y.y +=spdata.y*vecdata.x + spdata.x*vecdata.y;
     }
     //LOCAL_BARRIER;
        out[myRow]= y;
    }
    };        
  
  
  
    
        
    
"""
# from numpy import unsigned int32
# scalar_arg_dtypes=[unsigned int32, None, None, None, None, None]        
