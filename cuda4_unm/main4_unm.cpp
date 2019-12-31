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
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "pic_type.h"

// Function prototype from .cu file
void cu_run_grayscale( CudaPic t_bgr_pic, CudaPic t_bw_pic );

int main( int t_numarg, char **t_arg )
{
	// Uniform Memory allocator for Mat
	UniformAllocator allocator;
	cv::Mat::setDefaultAllocator( &allocator );

	if ( t_numarg < 2 )
	{
		printf( "Enter picture filename!\n" );
		return 1;
	}

	// Load image
	cv::Mat l_cv_bgr = cv::imread( t_arg[ 1 ], CV_LOAD_IMAGE_COLOR );

	if ( !l_cv_bgr.data )
	{
		printf( "Unable to read file '%s'\n", t_arg[ 1 ] );
		return 1;
	}

	// create empty BW image
	cv::Mat l_cv_bw( l_cv_bgr.size(), CV_8U );

	// data for CUDA
	CudaPic l_pic_bgr, l_pic_bw;
	l_pic_bgr.m_size.x = l_pic_bw.m_size.x = l_cv_bgr.size().width;
	l_pic_bgr.m_size.y = l_pic_bw.m_size.y = l_cv_bgr.size().height;
	l_pic_bgr.m_p_uchar3 = ( uchar3 * ) l_cv_bgr.data;
	l_pic_bw.m_p_uchar1 = ( uchar1 * ) l_cv_bw.data;

	// Function calling from .cu file
	cu_run_grayscale( l_pic_bgr, l_pic_bw );

	// Show the Color and BW image
	cv::imshow( "Color", l_cv_bgr );
	cv::imshow( "GrayScale", l_cv_bw );
	cv::waitKey( 0 );
}

