// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Image transformation from RGB to BW schema. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "pic_type.h"

// Demo kernel to transform RGB color schema to BW schema
__global__ void kernel_grayscale( CudaPic t_color_pic, CudaPic t_bw_pic )
{
	// X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_color_pic.m_size.y ) return;
	if ( l_x >= t_color_pic.m_size.x ) return;

	// Get point from color picture
	uchar3 l_bgr = t_color_pic.m_p_uchar3[ l_y * t_color_pic.m_size.x + l_x ];

	// Store BW point to new image
	t_bw_pic.m_p_uchar1[ l_y * t_bw_pic.m_size.x + l_x ].x = l_bgr.x * 0.11 + l_bgr.y * 0.59 + l_bgr.z * 0.30;
}

void cu_run_grayscale( CudaPic t_color_pic, CudaPic t_bw_pic )
{
	cudaError_t l_cerr;

	// Grid creation, size of grid must be equal or greater than images
	int l_block_size = 16;
	dim3 l_blocks( ( t_color_pic.m_size.x + l_block_size - 1 ) / l_block_size, ( t_color_pic.m_size.y + l_block_size - 1 ) / l_block_size );
	dim3 l_threads( l_block_size, l_block_size );
	kernel_grayscale<<< l_blocks, l_threads >>>( t_color_pic, t_bw_pic );

	if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

	cudaDeviceSynchronize();
}
