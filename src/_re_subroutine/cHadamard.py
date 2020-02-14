"""
cHadamard
"""

R="""

KERNEL void cSelect2(
    const unsigned int Reps, 
    GLOBAL_MEM const  unsigned int *order1,
    GLOBAL_MEM const  unsigned int *order2,
    GLOBAL_MEM const float2 *indata,
    GLOBAL_MEM       float2 *outdata)
{
const unsigned int gid=get_global_id(0); 
const unsigned int t = ((float)gid / (float)Reps); // indptr 
const unsigned int r = gid - t*Reps; // residue

const unsigned int index2 = order2[t]*Reps + r;
const unsigned int index1 = order1[t]*Reps + r;
 
outdata[index2]=indata[index1];
};

KERNEL void cDistribute(    
      const    unsigned int    Reps,
      const    unsigned int    prodNd, 
      GLOBAL_MEM const float2 *arr_large, // sensitivity and scaling array, prodNd*Reps
      GLOBAL_MEM const float2 *arr_image, // image array, prodNd
      GLOBAL_MEM float2 *arr_out // output array, prodNd * Reps
      ) 
{  
    const unsigned int t = get_global_id(0);
    //const unsigned int nd = t/Reps;
    const unsigned int nd = ((float)t / (float)Reps);
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
    const    unsigned int    Reps,
    const    unsigned int    prodNd, 
    GLOBAL_MEM const float2 *arr_large, // sensitivity and scaling array, prodNd*Reps
    GLOBAL_MEM const float2 *arr_image, // image array, prodNd*Reps
    GLOBAL_MEM float2 *arr_out // reduced output array, prodNd 
    )
{   
    const unsigned int t = get_local_id(0);
    //const float float_reps = (float)Reps;
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
    // float2  y= zero;
    if (myRow < prodNd)
    {
     const unsigned int vecStart = myRow * Reps;
     const unsigned int vecEnd = vecStart + Reps;            
     for (unsigned int j = vecStart+id;  j<vecEnd; j += vecWidth)
     {
          
            const float2 u = arr_large[j]; // sensitivities and scaling, complex 
            const float2 v = arr_image[j];  
            float2 w; 
            w.x = u.x * v.x + u.y * v.y; // conjugate of u 
            w.y = u.x * v.y - u.y * v.x; 
            w.x = w.x/(float)Reps;
            w.y = w.y/(float)Reps;
          partialSums[t] = w;        //partialSums[t] + y;
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
       arr_out[myRow]=partialSums[t]; 
      }
     }
};    // End of cMerge
  
KERNEL void cAggregate(
    const    unsigned int    Reps,
    const    unsigned int    prodNd, 
    GLOBAL_MEM const float2 *arr_image, // image array, prodNd*Reps
    GLOBAL_MEM float2 *arr_out // reduced output array, prodNd 
    )
{   
    const unsigned int t = get_local_id(0);
    //const float float_reps = (float)Reps;
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
    // float2  y= zero;
    if (myRow < prodNd)
    {
     const unsigned int vecStart = myRow * Reps;
     const unsigned int vecEnd = vecStart + Reps;            
     for (unsigned int j = vecStart+id;  j<vecEnd; j += vecWidth)
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
       arr_out[myRow]=partialSums[t]; 
      }
     }
     LOCAL_BARRIER;
};    // End of cAggregate
    
KERNEL void cPopulate(    
      const    unsigned int    Reps,
      const    unsigned int   prodNd, 
      GLOBAL_MEM const float2 *arr_image, // image array, prodNd
      GLOBAL_MEM float2 *arr_out // output array, prodNd * Reps
      ) 
{  
    const unsigned int t = get_global_id(0);
    //const unsigned int nd = t/Reps;
    const unsigned int nd = ((float)t / (float)Reps);
    if (nd < prodNd){
    const float2 v = arr_image[nd];
    arr_out[t] = v;
    }
    LOCAL_BARRIER;
};  // End of cPopulate    
    
"""
# from numpy import unsigned int32
# scalar_arg_dtypes=[unsigned int32, None, None, None, None, None]        
