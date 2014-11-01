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
* fast_tracking.cu
* 
* Author: Ilker GURCAN
*/

#define HETLD_EXPORTS

#include "hetld_errors.hpp"
#include "utilities.hpp"
#include "fast_tracking.cuh"

using namespace cv::gpu;

__host__
FastTracking::FastTracking(MemoryManagement *mem_module) {

	_mem_module = mem_module;
}

__host__
void FastTracking::initializeTLDObject(FAST_TRACKING_STR *fast_tr, 
                                       bool use_initial_flows, 
									   cv::Size win_size, 
									   int iters,
									   int max_pyr_level) {

	if(fast_tr == NULL)
		throw HETLDError(format(GETFTRERROR(ER_FASTTR_TLD_TR_OBJ_NULL),
			                    "initializeTLDObject"),
						 fast_tracking,
						 ER_FASTTR_TLD_TR_OBJ_NULL);
	fast_tr->lk                  = new PyrLKOpticalFlow();
	fast_tr->lk->useInitialFlow  = use_initial_flows;
	fast_tr->lk->winSize.width   = win_size.width;
	fast_tr->lk->winSize.height  = win_size.height;
	fast_tr->lk->iters           = iters;
	fast_tr->lk->maxLevel        = max_pyr_level;
}

__host__
void FastTracking::finalizeTLDObject(FAST_TRACKING_STR *fast_tr) {

	if(fast_tr->lk != nullptr)
		delete fast_tr->lk;
}

//TODO Check out Whether Input & Output Parameters are Valid!...
__host__
int FastTracking::runFBOpticalFlows(FAST_TRACKING_STR *fast_tr,
	                                int num_points,
									uchar *out_lk_status,
									uchar *out_fb_status,
									cv::Point2f *pts_prev,
									cv::Point2f *out_pts_cur,
									cv::Point2f *out_fb_pts) {
	
	if(fast_tr == NULL)
		throw HETLDError(format(GETFTRERROR(ER_FASTTR_TLD_TR_OBJ_NULL),
			                    "runFBOpticalFlows"),
						 fast_tracking,
						 ER_FASTTR_TLD_TR_OBJ_NULL);
	//For Timing...
	__int64 counter_start; 
	double pc_freq;
	double total_time;
	//***********************************************************************
	//*Upload Data to GPU Before Computing Forward-Backward Optical Flows...*
	//***********************************************************************
	GpuMat prev_img_hook(_mem_module->getSROI()->height,
		                 _mem_module->getSROI()->width,
						 CV_8UC1,
						 (void*)_mem_module->getDevPrevFrame(),
						 (size_t)_mem_module->getDROI()->width);//Row Length in Bytes(Including Padding)...
	GpuMat cur_img_hook(_mem_module->getSROI()->height,
		                _mem_module->getSROI()->width,
						CV_8UC1,
						(void*)_mem_module->getDevCurFrame(),
						(size_t)_mem_module->getDROI()->width);
	//For status...
	cv::Mat status(1, 
		           num_points, 
				   CV_8UC1, 
				   (void*)out_lk_status);
	cv::Mat fb_status(1, 
					  num_points,
				      CV_8UC1,
					  (void*)out_fb_status);
	GpuMat dev_status;
	GpuMat dev_fb_status;
	//For pts...
	cv::Mat prev_pts(1, 
		             num_points, 
					 CV_32FC2, 
					 (void*)pts_prev);
	cv::Mat cur_pts(1, 
		            num_points, 
					CV_32FC2, 
					(void*)out_pts_cur);
	cv::Mat fb_pts(1,
				   num_points, 
				   CV_32FC2,
				   (void*)out_fb_pts);
	GpuMat dev_prev_pts;
	GpuMat dev_cur_pts;
	GpuMat dev_fb_pts;
	
	dev_prev_pts.upload(prev_pts);
	dev_cur_pts.upload(cur_pts);
	dev_fb_pts.upload(fb_pts);
	//***********************************************************************
	//****************Apply Lucas-Kanade in Both Directions...***************
	//***********************************************************************
	startCounter(&counter_start, &pc_freq);
	//In Forward Direction...
	fast_tr->lk->sparse(prev_img_hook,
					    cur_img_hook,
						dev_prev_pts,
						dev_cur_pts,
						dev_status);
	//In Reverse Direction...
	fast_tr->lk->sparse(cur_img_hook,
						prev_img_hook,
						dev_cur_pts,
						dev_fb_pts,
						dev_fb_status);
	total_time = getCounter(&counter_start, &pc_freq);
	dev_cur_pts.download(cur_pts);
	dev_status.download(status);
	dev_fb_pts.download(fb_pts);
	dev_fb_status.download(fb_status);
	//Eliminate All Points Which've Failed One or Both of the Optical Flow Tests...
	for(int i = 0; i<num_points; i++)
		out_fb_status[i] &= out_lk_status[i];
	return total_time;
}

__host__
FastTracking::~FastTracking() {
	
}
