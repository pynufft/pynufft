"""
==============================================
Conjugate transpose of spmv.
Offload the conjugate transpose of Sparse Matrix Vector Multiplication to heterogeneous devices.
Note: Now pyopencl/pycuda are supported. 
"""

R="""
KERNEL void cCSR_spmvh_scalar(   
        const    unsigned int    n_row,
        GLOBAL_MEM    const    unsigned int    *indptr, 
        GLOBAL_MEM    const    unsigned int    *indices,
        GLOBAL_MEM    const    float2    *data,
        GLOBAL_MEM    float    *kx, 
        GLOBAL_MEM    float    *ky,
        GLOBAL_MEM   const    float2    *Yx)
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
        for (unsigned int j = vecStart + 0; j < vecEnd;  j += 1) //vecWidth)
        {
            const unsigned int col = indices[j];
            const float2 spdata = data[j]; // row
            u.x =  spdata.x * y.x + spdata.y * y.y;
            u.y =  - spdata.y * y.x + spdata.x * y.y;
            atomic_add_float(kx + col, u.x);
            atomic_add_float(ky + col, u.y);
        }; // for unsigned int j 
    }; //  if (myRow < n_row)
}; // Program entry
                
KERNEL void cELL_spmvh_scalar(   
        const    unsigned int    n_row,
        GLOBAL_MEM    const    unsigned int    *indptr, 
        GLOBAL_MEM    const    unsigned int    *indices,
        GLOBAL_MEM    const    float2    *data,
        GLOBAL_MEM    float    *kx, 
        GLOBAL_MEM    float    *ky,
        GLOBAL_MEM   const    float2    *Yx)
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
        for (unsigned int j = vecStart + 0; j < vecEnd;  j += 1) //vecWidth)
        {
            const unsigned int col = indices[j];
            const float2 spdata = data[j]; // row
            u.x =  spdata.x * y.x + spdata.y * y.y;
            u.y =  - spdata.y * y.x + spdata.x * y.y;
            atomic_add_float(kx + col, u.x);
            atomic_add_float(ky + col, u.y);
        }; // for unsigned int j 
    }; //  if (myRow < n_row)
}; // Program entry    
    
KERNEL void pELL_spmvh_scalar(    
        const    unsigned int    nRow,        // number of rows
        const    unsigned int    prodJd,     // product of Jd
        const    unsigned int    sumJd,     // sum of Jd
        const    unsigned int    dim,           // dimensionality
        GLOBAL_MEM const unsigned int *Jd,    // Jd
        // GLOBAL_MEM const unsigned int *curr_sumJd,    // 
        GLOBAL_MEM const unsigned int *meshindex,    // meshindex, prodJd * dim
        GLOBAL_MEM const unsigned int *kindx,    // unmixed column indexes of all dimensions
        GLOBAL_MEM const float2 *udata,    // interpolation data before Kronecker product
        GLOBAL_MEM    float    *kx, 
        GLOBAL_MEM    float    *ky,
        GLOBAL_MEM const float2 *input)   // y
{      
    unsigned int myRow= get_global_id(0);
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    if (myRow < nRow)
    {
         float2 u = zero;
         for (unsigned int j = 0;  j  <  prodJd; j ++)
         {    
            // now doing the first dimension
            unsigned int index_shift = myRow * sumJd;
            // unsigned int tmp_sumJd = 0;
            unsigned int J = Jd[0];
            unsigned int index =    index_shift +  meshindex[dim*j + 0];
            unsigned int col = kindx[index] ;
            float2 spdata = udata[index];
            index_shift += J; 
            for (unsigned int dimid = 1; dimid < dim; dimid ++ )
            {
                J = Jd[dimid];
                index =   index_shift + meshindex[dim*j + dimid];   // the index of the partial ELL arrays *kindx and *udata
                col += kindx[index] + 1  ;                                            // the column index of the current j
                float tmp_x = spdata.x;
                float2 tmp_udata = udata[index];
                spdata.x = spdata.x * tmp_udata.x - spdata.y * tmp_udata.y;                            // the spdata of the current j
                spdata.y = tmp_x * tmp_udata.y + spdata.y * tmp_udata.x; 
                index_shift  += J;
            }; // Iterate over dimensions 1 -> Nd - 1
            
            float2 ydata=input[myRow]; // kout[col];
            u.x =  spdata.x*ydata.x + spdata.y*ydata.y;
            u.y =  - spdata.y*ydata.x + spdata.x*ydata.y;
            
            atomic_add_float(kx + col, u.x);
            atomic_add_float(ky + col, u.y);
           
        }; // Iterate for (unsigned int j = 0;  j  <  prodJd; j ++)
    };  // if (myRow < nRow)
};    // End of pELL_spmv_scalar    
    

KERNEL void pELL_spmvh_vector(    
        const    unsigned int    nRow,        // number of rows
        const    unsigned int    prodJd,     // product of Jd
        const    unsigned int    sumJd,     // sum of Jd
        const    unsigned int    dim,           // dimensionality
        GLOBAL_MEM const unsigned int *Jd,    // Jd
        // GLOBAL_MEM const unsigned int *curr_sumJd,    // 
        GLOBAL_MEM const unsigned int *meshindex,    // meshindex, prodJd * dim
        GLOBAL_MEM const unsigned int *kindx,    // unmixed column indexes of all dimensions
        GLOBAL_MEM const float2 *udata,    // interpolation data before Kronecker product
        GLOBAL_MEM    float    *kx, 
        GLOBAL_MEM    float    *ky,
        GLOBAL_MEM const float2 *input)   // y
{      
    const unsigned int t = get_local_id(0);
    const unsigned int d = get_local_size(0);
    const unsigned int g = get_group_id(0);
    const unsigned int a = g/prodJd;
    const unsigned int b = g - (prodJd*a);
    const unsigned int myRow = a*d + t;
    const unsigned int j = b;
    
    // unsigned int myRow= get_global_id(0);
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    if (myRow < nRow)
    {
         float2 u = zero;
       //  for (unsigned int j = 0;  j  <  prodJd; j ++)
       if (j<prodJd)
         {    
            // now doing the first dimension
            unsigned int index_shift = myRow * sumJd;
            // unsigned int tmp_sumJd = 0;
            unsigned int J = Jd[0];
            unsigned int index =    index_shift +  meshindex[dim*j + 0];
            unsigned int col = kindx[index] ;
            float2 spdata = udata[index];
            index_shift += J; 
            for (unsigned int dimid = 1; dimid < dim; dimid ++ )
            {
                J = Jd[dimid];
                index =   index_shift + meshindex[dim*j + dimid];   // the index of the partial ELL arrays *kindx and *udata
                col += kindx[index] + 1  ;                                            // the column index of the current j
                float tmp_x = spdata.x;
                float2 tmp_udata = udata[index];
                spdata.x = spdata.x * tmp_udata.x - spdata.y * tmp_udata.y;                            // the spdata of the current j
                spdata.y = tmp_x * tmp_udata.y + spdata.y * tmp_udata.x; 
                index_shift  += J;
            }; // Iterate over dimensions 1 -> Nd - 1
            
            float2 ydata=input[myRow]; // kout[col];
            u.x =  spdata.x*ydata.x + spdata.y*ydata.y;
            u.y =  - spdata.y*ydata.x + spdata.x*ydata.y;
            
            atomic_add_float(kx + col, u.x);
            atomic_add_float(ky + col, u.y);
           
      }; // Iterate for (unsigned int j = 0;  j  <  prodJd; j ++)
    };  // if (myRow < nRow)
    
};    // End of pELL_spmv_vector  
    

KERNEL void pELL_spmvh_mCoil(
        const    unsigned int    Reps,             // number of coils
        const    unsigned int    nRow,        // number of rows
        const    unsigned int    prodJd,     // product of Jd
        const    unsigned int    sumJd,     // sum of Jd
        const    unsigned int    dim,           // dimensionality
        GLOBAL_MEM const unsigned int *Jd,    // Jd
        // GLOBAL_MEM const unsigned int *curr_sumJd,    // 
        GLOBAL_MEM const unsigned int *meshindex,    // meshindex, prodJd * dim
        GLOBAL_MEM const unsigned int *kindx,    // unmixed column indexes of all dimensions
        GLOBAL_MEM const float2 *udata,    // interpolation data before Kronecker product
        GLOBAL_MEM    float    *kx, 
        GLOBAL_MEM    float    *ky,
        GLOBAL_MEM const float2 *input)   // y
{      
    const unsigned int t = get_local_id(0);
    const unsigned int d = get_local_size(0);
    const unsigned int g = get_group_id(0);
    const unsigned int a = g/prodJd;
    const unsigned int b = g - (prodJd*a);
    const unsigned int myRow = a*d + t;
    const unsigned int j = b;
    unsigned int m = myRow / Reps;
    unsigned int nc = myRow - m * Reps;
    
    
    // unsigned int myRow= get_global_id(0);
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    if (m< nRow)
    {    if (nc < Reps)
    {
         float2 u = zero;
       //  for (unsigned int j = 0;  j  <  prodJd; j ++)
       if (j<prodJd)
         {    
            // now doing the first dimension
            unsigned int index_shift = m * sumJd;
            // unsigned int tmp_sumJd = 0;
            unsigned int J = Jd[0];
            unsigned int index =    index_shift +  meshindex[dim*j + 0];
            unsigned int col = kindx[index] ;
            float2 spdata = udata[index];
            index_shift += J; 
            for (unsigned int dimid = 1; dimid < dim; dimid ++ )
            {
                J = Jd[dimid];
                index =   index_shift + meshindex[dim*j + dimid];   // the index of the partial ELL arrays *kindx and *udata
                col += kindx[index] + 1  ;                                            // the column index of the current j
                float tmp_x = spdata.x;
                float2 tmp_udata = udata[index];
                spdata.x = spdata.x * tmp_udata.x - spdata.y * tmp_udata.y;                            // the spdata of the current j
                spdata.y = tmp_x * tmp_udata.y + spdata.y * tmp_udata.x; 
                index_shift  += J;
            }; // Iterate over dimensions 1 -> Nd - 1
            
            float2 ydata=input[myRow]; // kout[col];
            u.x =  spdata.x*ydata.x + spdata.y*ydata.y;
            u.y =  - spdata.y*ydata.x + spdata.x*ydata.y;
            
            atomic_add_float(kx + col*Reps + nc, u.x);
            atomic_add_float(ky + col*Reps + nc, u.y);
           
      }; // Iterate for (unsigned int j = 0;  j  <  prodJd; j ++)
    }; // if (nc < Reps)
    };  // if (m < nRow)
    
};    // End of pELL_spmv_mCoil  
        
"""
# from numpy import unsigned int32
# scalar_arg_dtypes=[unsigned int32, None, None, None, None, None]        
