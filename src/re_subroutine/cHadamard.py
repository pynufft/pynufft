"""
cHadamard
"""

R="""

KERNEL void cSelect2(
    const uint Reps, 
    GLOBAL_MEM const  uint *order1,
    GLOBAL_MEM const  uint *order2,
    GLOBAL_MEM const float2 *indata,
    GLOBAL_MEM       float2 *outdata)
{
const uint gid=get_global_id(0); 
const uint t = ((float)gid / (float)Reps); // indptr 
const uint r = gid - t*Reps; // residue

const uint index2 = order2[t]*Reps + r;
const uint index1 = order1[t]*Reps + r;
 
outdata[index2]=indata[index1];
};

KERNEL void cDistribute(    
      const    uint    Reps,
      const    uint    prodNd, 
      GLOBAL_MEM const float2 *arr_large, // sensitivity and scaling array, prodNd*Reps
      GLOBAL_MEM const float2 *arr_image, // image array, prodNd
      GLOBAL_MEM float2 *arr_out // output array, prodNd * Reps
      ) 
{  
    const uint t = get_global_id(0);
    //const uint nd = t/Reps;
    const uint nd = ((float)t / (float)Reps);
    if (nd < prodNd){
    const float2 u = arr_large[t]; 
    const float2 v = arr_image[nd];
    float2 w; 
    w.x = u.x * v.x - u.y * v.y; 
    w.y = u.x * v.y + u.y * v.x; 
    arr_out[t] = w;
    }
};  // End of cDistribute



KERNEL void cMerge(
    const    uint    Reps,
    const    uint    prodNd, 
    GLOBAL_MEM const float2 *arr_large, // sensitivity and scaling array, prodNd*Reps
    GLOBAL_MEM const float2 *arr_image, // image array, prodNd*Reps
    GLOBAL_MEM float2 *arr_out // reduced output array, prodNd 
    )
{   
    const uint t = get_local_id(0);
    //const float float_reps = (float)Reps;
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
    // float2  y= zero;
    if (myRow < prodNd)
    {
     const uint vecStart = myRow * Reps;
     const uint vecEnd = vecStart + Reps;            
     for (uint j = vecStart+id;  j<vecEnd; j += vecWidth)
     {
          
            const float2 u = arr_large[j]; // sensitivities and scaling, complex 
            const float2 v = arr_image[j];  
            float2 w; 
            w.x = u.x * v.x + u.y * v.y; 
            w.y = u.x * v.y - u.y * v.x; 
            w.x = w.x/(float)Reps;
            w.y = w.y/(float)Reps;
          partialSums[t] = w;        //partialSums[t] + y;
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
       arr_out[myRow]=partialSums[t]; 
      }
     }
};    // End of cMerge
  
KERNEL void cAggregate(
    const    uint    Reps,
    const    uint    prodNd, 
    GLOBAL_MEM const float2 *arr_image, // image array, prodNd*Reps
    GLOBAL_MEM float2 *arr_out // reduced output array, prodNd 
    )
{   
    const uint t = get_local_id(0);
    //const float float_reps = (float)Reps;
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
    // float2  y= zero;
    if (myRow < prodNd)
    {
     const uint vecStart = myRow * Reps;
     const uint vecEnd = vecStart + Reps;            
     for (uint j = vecStart+id;  j<vecEnd; j += vecWidth)
     {
        float2 v = arr_image[j];
        v.x = v.x/(float)Reps;
        v.y = v.y/(float)Reps;
        partialSums[t] =partialSums[t] + v;        //partialSums[t] + y;
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
       arr_out[myRow]=partialSums[t]; 
      }
     }
};    // End of cAggregate
    
KERNEL void cPopulate(    
      const    uint    Reps,
      const    uint   prodNd, 
      GLOBAL_MEM const float2 *arr_image, // image array, prodNd
      GLOBAL_MEM float2 *arr_out // output array, prodNd * Reps
      ) 
{  
    const uint t = get_global_id(0);
    //const uint nd = t/Reps;
    const uint nd = ((float)t / (float)Reps);
    if (nd < prodNd){
    const float2 v = arr_image[nd];
    arr_out[t] = v;
    }
};  // End of cPopulate    
    
"""
from numpy import uint32
scalar_arg_dtypes=[uint32, None, None, None, None, None]        
