/*
*
* Copyright 2013-2014 METU, Middle East Technical University, Informatics Institute
*
* This file is part of H-TLD.
*
* H-TLD is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* H-TLD is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along 
* with the one from Surrey University for TLD Algorithm developed by Zdenek Kalal.
* If not, see <http://www.gnu.org/licenses/>. 
* Please contact Alptekin TEMIZEL for more info about 
* licensing atemizel@metu.edu.tr.
*
*/

/*
* mem_management.cu
* 
* Author: Ilker GURCAN
*/

//TODO Don't Forget to Catch All Exceptions Properly...
#define HETLD_EXPORTS

#include<iostream>
#include<stdlib.h>
#include<nppi.h>
#include<device_launch_parameters.h>
#include<opencv2\gpu\stream_accessor.hpp>
#include "hetld_macros.hpp"
#include "hetld_errors.hpp"
#include "utilities.hpp"
#include "mem_management.cuh"

using namespace cv::gpu;

__host__
MemoryManagement::MemoryManagement(int f_width, 
                                   int f_height, 
								   double sigma) {

	cudaError cuda_status;
	cv::gpu::StreamAccessor accessor;

	//Initialize Module Variables...
	_s_roi           = (NppiSize*)malloc(sizeof(NppiSize));
	_s_roi->width    = f_width;
	_s_roi->height   = f_height;
	_d_roi           = (NppiSize*)malloc(sizeof(NppiSize));
	_d_roi->height   = f_height;
	_cv_cuda_stream  = accessor.getStream(_cv_stream);
	//Check Out http://docs.opencv.org/modules/imgproc/doc/filtering.html#Mat getGaussianKernel(int ksize, double sigma, int ktype) for Why We Add +1...
	_gaussian_filter_gpu = createGaussianFilter_GPU(CV_8UC1, 
		                                            cv::Size((int)(6 * sigma) + 1, (int)(6 * sigma) + 1), 
													sigma, 
													0.0,
													cv::BORDER_CONSTANT,
													cv::BORDER_CONSTANT);
	
	//Create an Aligned Data...
	_dev_cur_frame = nppiMalloc_8u_C1(f_width,
		                              f_height,
								      &_d_roi->width);
	if(_dev_cur_frame == NULL)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_CUDA_MALLOC),
			             "Current Frame",
						 "Constructor",
						 "nppiMalloc_8u_C1 Error"),
						 mem_management,
						 ER_MEM_MAN_CUDA_MALLOC);

	_dev_prev_frame = nppiMalloc_8u_C1(f_width,
	 	                               f_height,
									   &_d_roi->width);
	if(_dev_prev_frame == NULL)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_CUDA_MALLOC),
			             "Previous Frame",
						 "Constructor",
						 "nppiMalloc_8u_C1 Error"),
						 mem_management,
						 ER_MEM_MAN_CUDA_MALLOC);
	
	cuda_status = cudaMalloc((void**)(&_dev_blurred_cur_frame), 
							 sizeof(Npp8u) * f_width * f_height);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_CUDA_MALLOC), 
								"Blurred Current Frame", 
								"Constructor", 
								cudaGetErrorString(cuda_status)),
						 mem_management, 
						 ER_MEM_MAN_CUDA_MALLOC);
	//Allocate Other Required Resources...
	cuda_status = cudaMalloc((void **)(&_dev_ii),
		                     sizeof(Npp32s) * (_d_roi->width + 1) * (_d_roi->height + 1));
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_CUDA_MALLOC), 
								"Integral Image", 
								"Constructor", 
								cudaGetErrorString(cuda_status)), 
						 mem_management, 
						 ER_MEM_MAN_CUDA_MALLOC);
	
	cuda_status = cudaMalloc((void **)(&_dev_ii2),
			                 sizeof(Npp32s) * (_d_roi->width + 1) * (_d_roi->height + 1));
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_CUDA_MALLOC), 
								"Squared Integral Image", 
								"Constructor", 
								cudaGetErrorString(cuda_status)), 
						 mem_management, 
						 ER_MEM_MAN_CUDA_MALLOC);
	_host_ii  = (Npp32s*)malloc(sizeof(Npp32s) * f_width * f_height);
	_host_ii2 = (Npp32s*)malloc(sizeof(Npp32s) * f_width * f_height);
	//In Order to Allocate it Once Per Process...
	if(_s_roi->width != _d_roi->width) {
		cuda_status = cudaMalloc((void**)(&_dev_temp_frame_buf), 
								 sizeof(Npp8u) * f_width * f_height);
		if(cuda_status != cudaSuccess)
			throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_CUDA_MALLOC), 
								    "Temporary Frame Buffer", 
								    "Constructor", 
									cudaGetErrorString(cuda_status)),
							 mem_management, 
						     ER_MEM_MAN_CUDA_MALLOC);
	} else 
		_dev_temp_frame_buf = NULL;
}

__host__
void findKLPForChangingDSF(int total_load, 
                           int* num_blocks, 
						   int* num_threads_per_block) {

	cudaDeviceProp *dev_prop;
	int expected_num_of_blocks;

	dev_prop               = getDeviceProps(0);
	*num_threads_per_block = findOptimalNumOfTPB(0);
	expected_num_of_blocks = (int)ceil(total_load / (float)(*num_threads_per_block));
	//Some Resources Are Idle! Use As Much of Those Resources As You Can...
	if(expected_num_of_blocks < dev_prop->multiProcessorCount) {
		while(*num_threads_per_block > 64 && expected_num_of_blocks < dev_prop->multiProcessorCount) {
			(*num_threads_per_block) -= 32;
			expected_num_of_blocks    = (int)ceil(total_load / (float)(*num_threads_per_block));
		}//End of while-Loop...
		*num_blocks = expected_num_of_blocks;
	} else 
		*num_blocks = expected_num_of_blocks;
}

//********************************************************************************
//***************************Moving BBOX Offs to GPU...***************************
//********************************************************************************
//0-)num_of_bboxes, 
//1-)total_item_count, 
//2-)stride
//Constant Memory May Broadcast the Same Value to Multiple Thread Registers At Once!
//Don't Define Them As Anything Other Than Arrays... It is What CUDA Means By Symbol!
__constant__ int BBOX_INT_PARAMS[3];
__constant__ float ONE_OVER_NUM_BB_ATTRS[1];

__global__ void convertBBOXOffsToSoA(int* array_of_structures,
                                     int* structure_of_arrays) {
	
	int attr_index;
	int idx_to_soa;
	int i;
	
	for(i = blockIdx.x * blockDim.x + threadIdx.x;
		i < BBOX_INT_PARAMS[1];
		i += BBOX_INT_PARAMS[2]) {
		attr_index                      = i - floor(i * ONE_OVER_NUM_BB_ATTRS[0]) * NUM_OF_BBOX_ATTRS_ON_GPU;// i % NUM_OF_BBOX_ATTRS
		idx_to_soa                      = attr_index * BBOX_INT_PARAMS[0] + //Ptr to Start Index of the Current Attribute
										  floor(i * ONE_OVER_NUM_BB_ATTRS[0]);//Ptr to That Attribute of This BBOX
		structure_of_arrays[idx_to_soa] = array_of_structures[i];
	}//End of for-Loop...
}

__host__
void MemoryManagement::moveBBOXOffsToGPU(int *load_balanced_bb_offs, 
                                         int *dev_bbox_offs,
										 int num_of_bboxes) {

	cudaError cuda_status;
	int *d_offs;
	int num_blocks            = 0;
	int num_threads_per_block = 0;
	int total_load;
	int consts[3];
	float one_over_num_attrs;
	
	total_load  = NUM_OF_BBOX_ATTRS_ON_GPU * num_of_bboxes;
	cuda_status = cudaMalloc((void **)(&d_offs), sizeof(int) * total_load);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_CUDA_MALLOC), 
								"BBOX Offs", 
								"moveBBOXOffsToGPU", 
								cudaGetErrorString(cuda_status)), 
						 mem_management, 
						 ER_MEM_MAN_CUDA_MALLOC);
	
	cuda_status = cudaMemcpy((void*)d_offs, 
		                     load_balanced_bb_offs, 
						     sizeof(int) * total_load, 
						     cudaMemcpyHostToDevice);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_CUDA_MEM_CPY), 
								"BBOX Offs", "Host", "Device", 
								cudaGetErrorString(cuda_status)), 
						 mem_management, 
						 ER_MEM_MAN_CUDA_MEM_CPY);
	
	//Find Kernel Launch Parameters for High Occupancy...
	findKLPForChangingDSF(total_load, 
		                  &num_blocks, 
						  &num_threads_per_block);
	if(num_threads_per_block < 64 || num_blocks <= 0)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_KERNEL_EXEC_CONFIG_PR), 
		                        "convertBBOXOffsToSoA"), 
						 mem_management, 
						 ER_MEM_MAN_KERNEL_EXEC_CONFIG_PR);

	//Integer Constants...
	consts[0] = num_of_bboxes;
	consts[1] = total_load;
	consts[2] = num_blocks * num_threads_per_block;
	//Move it Without Casting!
	cuda_status = cudaMemcpyToSymbol(BBOX_INT_PARAMS, 
		                             consts, 
									 sizeof(int) * 3);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_CUDA_MEM_CPY), 
								"Integer Constants for Converting BBOX Offs to the Form of Structure of Arrays", 
								"Host", "Device",
								cudaGetErrorString(cuda_status)), 
						 mem_management, 
						 ER_MEM_MAN_CUDA_MEM_CPY);
	one_over_num_attrs = 1.0 / (float)NUM_OF_BBOX_ATTRS_ON_GPU;
	//Move it Without Casting!
	cuda_status = cudaMemcpyToSymbol(ONE_OVER_NUM_BB_ATTRS, 
		                             &one_over_num_attrs, 
								     sizeof(float));
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_CUDA_MEM_CPY), 
								"Float Constant", 
								"Host", "Device",
								cudaGetErrorString(cuda_status)), 
						 mem_management, 
						 ER_MEM_MAN_CUDA_MEM_CPY);
	//Now Move Them to GPU and Store Them in SoA Fashion...
	convertBBOXOffsToSoA<<<num_blocks, num_threads_per_block>>>(d_offs, dev_bbox_offs);
	cudaDeviceSynchronize();
	cuda_status = cudaGetLastError();
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_CUDA_KER_FAILED), 
								"convertBBOXOffsToSoA", 
								cudaGetErrorString(cuda_status)), 
						 mem_management, 
						 ER_MEM_MAN_CUDA_KER_FAILED);
	//Release Device Memory...
	cudaFree((void*)d_offs);
}

//********************************************************************************
//*************************Moving Forests Offs to GPU...**************************
//********************************************************************************
__host__
void MemoryManagement::moveForestsOffsToGPU(int *forests_offs, 
		                                    int *dev_forests_offs, 
										    int  num_of_scale_levels,
										    int  num_of_trees,
										    int  num_of_features) {

	cudaError cuda_status;
	cuda_status = cudaMemcpy((void*)dev_forests_offs,
							 (void*)forests_offs,
							 num_of_scale_levels * num_of_trees * num_of_features * 2 * sizeof(int),
							 cudaMemcpyHostToDevice);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_CUDA_MEM_CPY), 
								"Forests Offs", "Host", "Device", 
								cudaGetErrorString(cuda_status)), 
						 mem_management, 
						 ER_MEM_MAN_CUDA_MEM_CPY);
}

//********************************************************************************
//***************************Moving Cur-Frame to GPU...***************************
//********************************************************************************
__host__
void MemoryManagement::moveCurFrameToGPU(ubyte* cur_frame, 
                                         bool is_keep_prev_on_gpu,
										 bool is_gen_blurred_image) {

	cudaError cuda_status;
	NppStatus npp_status;
	Npp8u *swap;
	
	if(_dev_cur_frame == NULL) 
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_DEV_PTR_NULL), 
		                        "Current Frame"), 
						 mem_management, 
						 ER_MEM_MAN_DEV_PTR_NULL);
	//***********************************************************************
	//***Move Previous to Its Memory Location; Before Copying New Frame...***
	//***********************************************************************
	if(is_keep_prev_on_gpu) {
		if(_dev_prev_frame == NULL) 
			throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_DEV_PTR_NULL), 
									"Prev Frame"), 
							 mem_management, 
							 ER_MEM_MAN_DEV_PTR_NULL);
		swap            = _dev_cur_frame;
		_dev_cur_frame  = _dev_prev_frame;
		_dev_prev_frame = swap;
	}//End of Outermost if-Block...
	//***********************************************************************
	//**********************Move Current Frame to GPU...*********************
	//***********************************************************************
	if(_dev_temp_frame_buf != NULL) {
		//Copy Data to a Temporary Buffer in Order to Run NPPI Routine to 
		//Copy it to Its Further Data Aligned Location...
		cuda_status = cudaMemcpy((void*)_dev_temp_frame_buf,
								 (void*)cur_frame,
							     sizeof(Npp8u) * _s_roi->width * _s_roi->height,
								 cudaMemcpyHostToDevice);
		if(cuda_status != cudaSuccess)
			throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_CUDA_MEM_CPY), 
							        "Temporary Buffer", "Host", "Device", 
									cudaGetErrorString(cuda_status)), 
						     mem_management, 
							 ER_MEM_MAN_CUDA_MEM_CPY);
		if(is_gen_blurred_image)
			genBlurredFrame(_dev_temp_frame_buf);
		//Copy it Further to Its Final Destination...
		npp_status = nppiCopyConstBorder_8u_C1R(_dev_temp_frame_buf,
											    sizeof(Npp8u) * _s_roi->width,
												*(_s_roi),
												_dev_cur_frame,
												sizeof(Npp8u) * _d_roi->width,
											    *(_d_roi),
											    0,
												0,
												(Npp8u)0);
		if(npp_status != NPP_SUCCESS)
			throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_NPPI_MEM_CPY), 
							        "Current Frame", "Device", "Device", 
									(int)npp_status), 
						     mem_management, 
							 ER_MEM_MAN_NPPI_MEM_CPY);
	} else {
		cuda_status = cudaMemcpy((void*)_dev_cur_frame, 
								 (void*)cur_frame, 
								 sizeof(Npp8u) * _s_roi->width * _s_roi->height, 
								 cudaMemcpyHostToDevice);
		if(cuda_status != cudaSuccess)
			throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_CUDA_MEM_CPY), 
							        "Temporary Buffer", "Host", "Device", 
									cudaGetErrorString(cuda_status)), 
						     mem_management, 
							 ER_MEM_MAN_CUDA_MEM_CPY);
		if(is_gen_blurred_image)
			genBlurredFrame(_dev_temp_frame_buf);
	}
	//Actually, It is not Necessary to Check Whether This Flag is On; 
	//Because There would be No Command Issued to This CUDA Stream; hence No Waiting
	//in the Case That Flag is Off...
	if(is_gen_blurred_image)
		_cv_stream.waitForCompletion();
}

__host__
void MemoryManagement::genBlurredFrame(Npp8u *dev_cur_frame) {

	GpuMat cur_frame_gpu(_s_roi->height,
						 _s_roi->width,
						 CV_8UC1,
						 (void*)dev_cur_frame);
	GpuMat blurred_cur_frame_gpu(_s_roi->height,
					             _s_roi->width,
								 CV_8UC1,
								 (void*)_dev_blurred_cur_frame);
	_gaussian_filter_gpu->apply(cur_frame_gpu,
						        blurred_cur_frame_gpu,
								cv::Rect(0, 0, cur_frame_gpu.cols, cur_frame_gpu.rows),
								_cv_stream);
}

//********************************************************************************
//*****************************Moving IIs to Host...******************************
//********************************************************************************
__host__
void MemoryManagement::moveIIsToHost() {

	cudaError cuda_status;
	NppStatus npp_status;
	Npp32s *temp_buf;
	Npp32s *src_offset;
	//***************************Error Check...
	if(_dev_ii == NULL)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_DEV_PTR_NULL), 
		                        "Device Ptr for II"), 
						 mem_management, 
						 ER_MEM_MAN_DEV_PTR_NULL);

	if(_dev_ii2 == NULL)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_DEV_PTR_NULL), 
		                        "Device Ptr for II2"), 
						 mem_management, 
						 ER_MEM_MAN_DEV_PTR_NULL);
	
	//************************************Start Copying...
	//Copy Back Only the Region of Interest; Because
	//We Have Two Different Types of Paddings Within Image Data on Device: 
	//One for the Sake of Data Alignment and 
	//One for Extra Top and Left Borders Generated Intentionally
	//By NPPI's SqrIntegral Method...
	
	//We Are not Copying Them Concurrently; Because There Had to Be 
	//An Additional Temporary Buffer to Be Used(Waste of Space)...
	//Plus, It Might Cause OutOfMemoryExceptions for Huge Frames...

	//Create Temporary Buffer in Order to Run NPPI Routine...
	cuda_status = cudaMalloc((void**)(&temp_buf), 
							 sizeof(Npp32s) * _s_roi->width * _s_roi->height);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_CUDA_MALLOC), 
								"Temporary Buffer", 
								"moveIIsToHost", 
							    cudaGetErrorString(cuda_status)), 
						 mem_management, 
						 ER_MEM_MAN_CUDA_MALLOC);
	//Copy Region of Interest to Its Final Destination on GPU for II(xROI = 1, yROI = 1)...
	src_offset = _dev_ii + (_d_roi->width + 1) + 1;
	npp_status = nppiCopy_32s_C1R(src_offset, 
		                          sizeof(Npp32s) * (_d_roi->width + 1), 
							      temp_buf, 
							      sizeof(Npp32s) * _s_roi->width, 
								  *(_s_roi));
	if(npp_status != NPP_SUCCESS)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_NPPI_MEM_CPY), 
							    "Integral Image", "Device", "Device", 
							    (int)npp_status), 
						 mem_management, 
						 ER_MEM_MAN_NPPI_MEM_CPY);
	//Now Copy II to Host...
	cuda_status = cudaMemcpy((void*)_host_ii,
		                     (void*)temp_buf,
						     sizeof(Npp32s) * _s_roi->width * _s_roi->height,
							 cudaMemcpyDeviceToHost);
	//Copy Region of Interest to Its Final Destination on GPU for II2(xROI = 1, yROI = 1)...
	src_offset = _dev_ii2 + (_d_roi->width + 1) + 1;
	npp_status = nppiCopy_32s_C1R(src_offset, 
		                          sizeof(Npp32s) * (_d_roi->width + 1), 
								  temp_buf, 
								  sizeof(Npp32s) * _s_roi->width, 
								  *(_s_roi));
	if(npp_status != NPP_SUCCESS)
		throw HETLDError(format(GETMEMERROR(ER_MEM_MAN_NPPI_MEM_CPY), 
							    "Integral Image 2", "Device", "Device", 
							    (int)npp_status), 
						 mem_management, 
						 ER_MEM_MAN_NPPI_MEM_CPY);
	//Now Copy II2 to Host...
	cuda_status = cudaMemcpy((void*)_host_ii2, 
		                     (void*)temp_buf,
						     sizeof(Npp32s) * _s_roi->width * _s_roi->height, 
							 cudaMemcpyDeviceToHost);
	//Release Temporary Buffer...
	cuda_status = cudaFree((void*)temp_buf);
	if(cuda_status != cudaSuccess)
		fprintf(stderr, 
				GETMEMERROR(ER_MEM_MAN_CUDA_FREE), 
				"Temporary Buffer", 
				"moveIIsToHost", 
				cudaGetErrorString(cuda_status));
}

__host__
MemoryManagement::~MemoryManagement() {

	cudaError cuda_status;

	free((void*)_s_roi);
	free((void*)_d_roi);

	//Deallocate All Allocated Resources...
	if(_dev_cur_frame != NULL)
		nppiFree((void*)_dev_cur_frame);
	if(_dev_prev_frame != NULL)
		nppiFree((void*)_dev_prev_frame);
	if(_dev_blurred_cur_frame != NULL) {
		cuda_status = cudaFree((void*)_dev_blurred_cur_frame);
		if(cuda_status != cudaSuccess) {
			fprintf(stderr,
				    GETMEMERROR(ER_MEM_MAN_CUDA_FREE),
					"Blurred Current Frame",
					"Destructor",
					cudaGetErrorString(cuda_status));
		}//End of Innermost-if-Block...
	}
	if(_dev_ii != NULL) {
		cuda_status = cudaFree((void*)_dev_ii);
		if(cuda_status != cudaSuccess) {
			fprintf(stderr, 
					GETMEMERROR(ER_MEM_MAN_CUDA_FREE), 
					"Integral Image", 
					"Destructor", 
					cudaGetErrorString(cuda_status));
		}//End of Innermost-if-Block...
	}
	
	if(_dev_ii2 != NULL) {
		cuda_status = cudaFree((void*)_dev_ii2);
		if(cuda_status != cudaSuccess) { 
			fprintf(stderr, 
				    GETMEMERROR(ER_MEM_MAN_CUDA_FREE), 
					"Squared Integral Image", 
					"Destructor", 
					cudaGetErrorString(cuda_status));
		}//End of Innermost-if-Block...
	}

	if(_dev_temp_frame_buf != NULL) {
		cuda_status = cudaFree((void*)_dev_temp_frame_buf);
		if(cuda_status != cudaSuccess) {
			fprintf(stderr,
				    GETMEMERROR(ER_MEM_MAN_CUDA_FREE),
					"Temporary Buffer",
					"Destructor",
					cudaGetErrorString(cuda_status));
		}//End of Innermost-if-Block...
	}
	
	if(_host_ii != NULL)
		free((void*)_host_ii);
	if(_host_ii2 != NULL)
		free((void*)_host_ii2);
}
