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
* mem_management.cuh
* 
* Author: Ilker GURCAN
*/

#ifndef MEMMANAGEMENT_CUH_
#define MEMMANAGEMENT_CUH_

#include<cuda_runtime_api.h>
#include<nppdefs.h>
#include<opencv2\core\core.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\gpu\gpu.hpp>
#include<math.h>
#include "imodule.hpp"

//Module Definition...
class HETLD_API MemoryManagement : public IModule
{
public:
	__host__
	MemoryManagement(int f_width, 
	                 int f_height, 
					 double sigma);
	__host__
	virtual ~MemoryManagement();
	__host__
	enum HETLDModules getModule() {return mem_management;};
	//API Methods...
	__host__
	void moveBBOXOffsToGPU(int *load_balanced_bb_offs, 
		                   int *dev_bbox_offs, 
						   int  num_of_bboxes);
	__host__
	void moveForestsOffsToGPU(int *forests_offs, 
		                      int *dev_forests_offs, 
							  int  num_of_scale_levels,
						      int  num_of_trees,
							  int  num_of_features);
	__host__
	void moveCurFrameToGPU(ubyte *cur_frame, 
	                       bool is_keep_prev_on_gpu,
						   bool is_gen_blurred_image);
	__host__
	void moveIIsToHost();
	//Getters...
	__host__
	NppiSize* getSROI() {return _s_roi;};
	__host__
	NppiSize* getDROI() {return _d_roi;};
	__host__
	Npp8u* getDevCurFrame() {return _dev_cur_frame;};
	__host__
	Npp8u* getDevPrevFrame() {return _dev_prev_frame;};
	__host__
	Npp8u* getDevBlurredCurFrame() {return _dev_blurred_cur_frame;};
	__host__
	Npp32s* getDevII() {return _dev_ii;};
	__host__
	Npp32s* getDevII2() {return _dev_ii2;};
	__host__
	Npp32s* getHostII() {return _host_ii;};
	__host__
	Npp32s* getHostII2() {return _host_ii2;};
private:
	NppiSize                          *_s_roi;
	NppiSize                          *_d_roi;
	Npp8u                             *_dev_temp_frame_buf;
	Npp8u                             *_dev_prev_frame;
	Npp8u                             *_dev_cur_frame;
	Npp8u                             *_dev_blurred_cur_frame;
	Npp32s                            *_dev_ii;
	Npp32s                            *_dev_ii2;
	Npp32s                            *_host_ii;
	Npp32s                            *_host_ii2;
	cv::Ptr<cv::gpu::FilterEngine_GPU> _gaussian_filter_gpu;/*Why Not Simple GaussianBlur : http://docs.opencv.org/modules/gpu/doc/image_filtering.html#gpu-filterengine-gpu */
	cv::gpu::Stream                    _cv_stream;// In order to Overlap Some OpenCV Related Ops...
	cudaStream_t                       _cv_cuda_stream;
	//Private Methods...
	__host__
	void genBlurredFrame(Npp8u *dev_cur_frame);
};

#endif /* MEMMANAGEMENT_CUH_ */
