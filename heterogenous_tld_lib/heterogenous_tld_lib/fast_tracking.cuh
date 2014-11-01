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
* fast_tracking.cuh
* 
* Author: Ilker GURCAN
*/

#pragma once

#include<vector>
#include<cuda_runtime_api.h>
#include<opencv2\opencv.hpp>
#include<opencv2\gpu\gpu.hpp>
#include "imodule.hpp"
#include "mem_management.cuh"

//Data Structure for Fast Detection on CPU/GPU Heterogeneously...
struct fast_tracking_str {

	//**********************************************************************
	//****************Forward-Backward Optical Flow Comp...*****************
	//**********************************************************************
	cv::gpu::PyrLKOpticalFlow *lk;
};

typedef struct fast_tracking_str FAST_TRACKING_STR;

class HETLD_API FastTracking : public IModule 
{

public:
	__host__
	FastTracking(MemoryManagement *mem_module);
	__host__
	enum HETLDModules getModule() {return fast_tracking;};
	__host__
	virtual	~FastTracking();
	__host__
	void initializeTLDObject(FAST_TRACKING_STR *fast_tr, 
	                         bool use_initial_flows,
							 cv::Size win_size,
							 int iters,
							 int max_pyr_level);
	__host__
	void finalizeTLDObject(FAST_TRACKING_STR *fast_tr);
	//API Methods...
	__host__
	int runFBOpticalFlows(FAST_TRACKING_STR *fast_tr, 
	                      int num_points,
						  uchar *out_lk_status,
						  uchar *out_fb_status,
						  cv::Point2f *pts_prev,
						  cv::Point2f *out_pts_cur,
						  cv::Point2f *out_fb_pts);
private:
	MemoryManagement *_mem_module;
};
