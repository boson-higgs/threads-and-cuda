// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
//
// Image interface for CUDA
//
// ***********************************************************************

#pragma once

// Structure definition for exchanging data between Host and Device
struct CudaPic
{
  uint3 m_size;				// size of picture
  union {
	  void   *m_p_void;		// data of picture
	  uchar1 *m_p_uchar1;	// data of picture
	  uchar3 *m_p_uchar3;	// data of picture
	  uchar4 *m_p_uchar4;	// data of picture
  };
};
