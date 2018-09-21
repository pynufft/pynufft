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

    
KERNEL void cELL_spmv_vector(    
      const    uint    numRow,
      const    uint    colWidth,
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
     const uint vecStart = myRow*colWidth; 
     const uint vecEnd = (myRow + 1)*colWidth;             
     for (uint j = vecStart+id;  j<vecEnd; j += vecWidth)
     {
          const uint col    =    cols[j];
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
      uint bar = vecWidth / 2;
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
      const    uint    nRow,
      const    uint    colWidth, 
      GLOBAL_MEM const uint *cols,
      GLOBAL_MEM const float2 *data,
      GLOBAL_MEM const float2 *vec, 
      GLOBAL_MEM float2 *out)
{      uint myRow= get_global_id(0);
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    if (myRow < nRow)
    {      
     float2  y= zero;
     out[myRow] =zero; 
     for (uint j = myRow *  colWidth;  j< (myRow + 1) *  colWidth; j ++)
     //for (uint j = 0;  j<   colWidth; j ++)
     {
          uint col = cols[j];
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
from numpy import uint32
scalar_arg_dtypes=[uint32, None, None, None, None, None]        
