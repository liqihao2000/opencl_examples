//
//  test.h
//  HelloCL_CPP
//
//  Created by QI LI on 2021/1/1.
//  Copyright © 2021 QI LI. All rights reserved.
//

#ifndef test_h
#define test_h


__kernel void hello_kernel(__global const int *a,
                           __global const int *b,
                           __global int *result,int  ha,
                           int  wb,int  wa)
{
//    int row = get_global_id(0);
//    //int col = get_global_id(1);
//
//
//    int sum = 0;
//
//
//    for(int j=0;j<wb;j++){
//        sum=0;
//        for(int i=0;i<wa;i++)
//        {
//            sum += a[row*wa+i] * b[i*wb+j];
//        }
//        result[row*wb+j] = sum;
//    }
    int i=get_global_id(0);
    result[i]=a[i]+b[i];
    
    
}
//__kernel void hello_kernel(__global const int *a,
//                           __global const int *b,
//                           __global float *result,int  ha,int  wa,
//                           int  wb)
//{
//    int row = get_global_id(1);//4
//    int col = get_global_id(0);//5
//
//
//    int sum = 0;
//    for(int i=0;i<wa;i++)
//    {
//        sum += a[row*wa+i] * b[i*wb+col];
//    }
//
//    result[row*wb+col] = 2;
//}
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

#define TS 16

__kernel void hello_kernel(__global const int *a,
                           __global const int *b,
                           __global int *result_matrix,int  result_matrix_row,
                           int  result_matrix_col,int  compute_size)
{
    //    __local int lo[10];
    //    int idx = get_global_id(0); // statement 1
    //    printf("started for %d workitem\n", idx); // statement 2
    //    lo[idx] = idx + 100; // statement 3
    //    printf("value changed to %lf in %d workitem\n", lo[idx], idx); // statement 4
    //    barrier(CLK_LOCAL_MEM_FENCE); // statement 5
    //    for(int i=0;i<10;i++){
    //        result_matrix[i]=lo[i];
    //    }
    //    printf("completed for %d workitem\n", idx); // statement 6    //result_matrix = c;
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
    
    // Local memory to fit a tile of TS*TS elements of A and B
    __local int Asub[TS][TS];
    __local int Bsub[TS][TS];
    
    // Initialise the accumulation register
    int acc = 0;
    
    //printf("Kk\n");
    // Loop over all tiles
    const int numTiles = compute_size/TS;
    for (int t=0; t<numTiles; t++) {
        
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col][row] = a[tiledCol*result_matrix_row + globalRow];
        Bsub[col][row] = b[globalCol*compute_size + tiledRow];
        // printf("col=%d,row=%d,t=%d\t",col,row,t);
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }
        //printf("L\t");
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store the final result in C
    result_matrix[globalCol*result_matrix_row + globalRow] = acc;
    
    
}


#endif /* test_h */
