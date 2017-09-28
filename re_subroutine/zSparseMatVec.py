R="""
KERNEL void zSparseMatVec(    const    uint    dim,
                    GLOBAL_MEM const uint *rowDelimiters, 
                    GLOBAL_MEM  const uint *cols,
                    GLOBAL_MEM  const double2 *val,
                    GLOBAL_MEM  const double2 *vec, 
                    GLOBAL_MEM  double2 *out)
{   
        const uint t = get_local_id(0);
        const uint vecWidth=LL;
        // Thread ID within wavefront
        const uint id = t & (vecWidth-1);
        // One row per wavefront
        uint vecsPerBlock = get_local_size(0) / vecWidth;
        uint myRow = (get_group_id(0) * vecsPerBlock) + (t / vecWidth);
        LOCAL_MEM  double2 partialSums[LL];
        double2 zero;
        zero.x=0.0;
        zero.y=0.0;
        partialSums[t] = zero;
        double2 mySum= zero;
        double2 sumk_err=zero;
        double2  y=zero;
        double2 sumk_s;
        double2 bp;
        if (myRow < dim)
        {
            const uint vecStart = rowDelimiters[myRow];
            const uint vecEnd = rowDelimiters[myRow+1];
            
            for (uint j= vecStart + id; j < vecEnd;  j+=vecWidth)
            {
                        const uint col = cols[j];
                        const   double2 spdata =  val[j];
                        const   double2 vecdata =  vec[col];
                        //y =${mul}( val[j],vec[col]);
                        y.x =  spdata.x*vecdata.x -  spdata.y*vecdata.y;
                        y.y =  spdata.y*vecdata.x + spdata.x*vecdata.y;
                        sumk_s = mySum+y;
                        bp = sumk_s - mySum;
                        sumk_err = sumk_err +  ((mySum - (sumk_s - bp)) + (y - bp));
                        mySum = sumk_s;
            }
           double2 new_error = zero;
                                                    
            y =sumk_err;
            sumk_s = mySum+y;
            bp = sumk_s - mySum;
            new_error = new_error + ((mySum - (sumk_s - bp)) + (y - bp));
            mySum = sumk_s;
                                                    
            partialSums[t] = mySum;
            barrier(CLK_LOCAL_MEM_FENCE);
                                                            
            // Reduce partial sums
            uint bar = vecWidth / 2;
            while(bar > 0)
            {
                if (id < bar)  // partialSums[t] += partialSums[t+ bar];
                {
                    y =partialSums[t+bar];
                    sumk_s = partialSums[t]+y;
                    bp = sumk_s - mySum;
                    new_error = new_error + ((partialSums[t] - (sumk_s - bp)) + (y - bp));
                    partialSums[t] = sumk_s;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                bar = bar / 2;
            }            
            // Write result 
            if (id == 0)
            {
                out[myRow] = partialSums[t] + new_error; 
            }
        }
      barrier(CLK_GLOBAL_MEM_FENCE);
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
#                                             GLOBAL_MEM const double2 *data,
#                                             GLOBAL_MEM const double2 *x, 
#                                            GLOBAL_MEM double2 *y)
# {  //LOCAL_MEM double2  *vals;
# const uint i = get_global_id(0);
#     if ( i < num_rows ){
#       double2 dot ;
#       dot.x=0.0;
#       dot.y=0.0;
#            int row_start = ptr[ i ];
#            int row_end = ptr[ i +1];
#            
#         double2 sumk_err;
#               sumk_err.x=0.0;
#               sumk_err.y=0.0;
#         double2 y2;
#             y2.x=0.0;
#             y2.y=0.0;
#         double2 sumk_s;
#         double2 bp;
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
#                double2 new_error ;
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