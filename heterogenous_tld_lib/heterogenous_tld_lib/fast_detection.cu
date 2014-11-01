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
* fast_detection.cu
* 
* Author: Ilker GURCAN
*/

//TODO Don't Forget to Catch All Exceptions Properly...
#define HETLD_EXPORTS

#include<math.h>
#include<omp.h>
#include<string.h>
#include<assert.h>
#include<device_launch_parameters.h>
#include<nppi.h>
#include<thrust\execution_policy.h>
#include<thrust\copy.h>
#include<cub\cub.cuh>
#include "hetld_macros.hpp"
#include "hetld_errors.hpp"
#include "utilities.hpp"
#include "fast_detection.cuh"

using namespace cub;

#define TH_REG_COUNT_FOR_PV_COMP (18 + 1) //+1 for Each Thread's Required Additional Registry...


//Integer-Typed Constants Per TLDObject...
//0-)total_num_of_bb
__constant__ int TLD_OBJ_INT_CONSTS[1];

//Integer-Typed Constants for PV Computation on GPU in Order: 
//0-)image_width,
//1-)integral_image_width,
__constant__ int PV_INT_CONSTS[2];

//Integer-Typed Constants for Flag Computation on GPU in Order: 
//0-)num_of_trees,
//1-)num_of_features,
//2-)[0] * [1] * num_of_scale_levels * 2 = Total Num of Feature Pts
__constant__ int RFIC_INT_CONSTS[3];

__host__
FastDetection::FastDetection(MemoryManagement *mem_module,
							 CPU_PROPS* cpu_props) {

	cudaError_t status;
	int consts[2];
	_mem_module         = mem_module;
	_num_of_cpu_threads = cpu_props->physical_proc_cores <= 2 ? 2 : cpu_props->physical_proc_cores;
	//Allocate Memory for Array of Pointers to IIs...
	Npp32s* iis[2];
	iis[0] = _mem_module->getDevII();
	iis[1] = _mem_module->getDevII2();
	status = cudaMalloc((void**)&_dev_iis, sizeof(Npp32s*) * 2);
	if(status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR),
								"II Array on GPU(Allocating)",
								cudaGetErrorString(status)),
						 fast_detection,
						 ER_FASTDET_CUDA_MEM_ERROR);
	status = cudaMemcpy((void*)_dev_iis, 
		                (void*)iis, 
						sizeof(Npp32s*) * 2, 
						cudaMemcpyHostToDevice);
	if(status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR),
							    "IIs Array(Copying)",
								cudaGetErrorString(status)),
					     fast_detection,
						 ER_FASTDET_CUDA_MEM_ERROR);
	//Move Constant Data to GPU for Computing PV...
	consts[0] = _mem_module->getSROI()->width;
	consts[1] = _mem_module->getDROI()->width + 1;
	status    = cudaMemcpyToSymbol(PV_INT_CONSTS, consts, sizeof(int) * 2);
	if(status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR), 
								"PV_INT_CONSTS (in constructor)", 
								cudaGetErrorString(status)), 
						 fast_detection, 
						 ER_FASTDET_CUDA_MEM_ERROR);
}

//********************************************************************************
//****************************Creating Clusters...********************************
//********************************************************************************
__host__
void findSMIndicesOfSLs(std::vector<SL_MAP*>& sl_with_sm, 
					    std::vector<SCAN_LINE_SET*>& scan_lines, 
						int cluster_size) {

	SCAN_LINE_SET* temp_sl;
	SCAN_LINE_SET::iterator sl_it;
	int i;
	int sm_idx, size;
	SL_MAP* temp_map;

	size = scan_lines.size();
	for(i = 0; i<size; i += cluster_size) {
		temp_sl  = scan_lines.at(i);
		temp_map = new SL_MAP();
		sm_idx   = -1;
		for(sl_it = temp_sl->begin();
			sl_it != temp_sl->end();
			++sl_it)
			temp_map->insert(SL_PAIR(*sl_it, ++sm_idx));
		sl_with_sm.push_back(temp_map);
	}//End of Outer-for-Loop...
}

__host__
void mergeScanLines(int cluster_size, 
                    std::vector<SCAN_LINE_SET*>& scan_lines) {

	SCAN_LINE_SET::iterator temp_it;
	SCAN_LINE_SET* cur;
	SCAN_LINE_SET* temp;
	int i, j, size;

	//Merge...
	size = scan_lines.size();
	for(i = 0; i<size; i += cluster_size) {
		cur = scan_lines.at(i);
		for(j = 1; 
			j<cluster_size && (i + j)<size; 
			j++) {
			temp = scan_lines.at(i + j);
			for(temp_it = temp->begin();
				temp_it != temp->end();
				++temp_it) {
				cur->insert(*temp_it);
			}//End of Innermost-for-Loop...
		}//End of Inner-for-Loop...
	}//End of Outermost-for-Loop...
}


__host__
void setScanLineOffs(int ii_width, 
                     int *start_offs,
					 SL_MAP *sl_map,
					 int left_corner, 
					 int right_corner, 
					 int num_of_bboxes) {

	int i, ii_row_idx, sm_row_idx;

	for(i = 0; i<num_of_bboxes; i++) {
		ii_row_idx               = start_offs[left_corner] / ii_width;//Find Row_Idx
		sm_row_idx               = sl_map->find(ii_row_idx)->second;
		start_offs[left_corner]  = sm_row_idx * ii_width + 
								   start_offs[left_corner] - ii_row_idx * ii_width;
		start_offs[right_corner] = sm_row_idx * ii_width + 
			                       start_offs[right_corner] - ii_row_idx * ii_width;
		start_offs              += NUM_OF_BBOX_ATTRS_ON_GPU;
	}//End of for-Loop...
}

__host__
void setSMIndicesofBBs(int *load_balanced_bbox_offs, 
                       std::vector<SL_MAP*>& sl_map_list,
					   int cluster_size,
					   int total_num_of_slp,
					   int ii_width, 
					   std::vector<int>& num_of_bb_per_slp,
					   host_vector<int>& cum_sum_of_bb_per_cl) {

	int i;
	int cluster_idx;
	int counter;
	int num_of_bboxes;

	cluster_idx             = 0;
	counter                 = cluster_size;
	num_of_bboxes           = 0;
	cum_sum_of_bb_per_cl[0] = 0;
	for(i = 0; i<total_num_of_slp; i++) {
		num_of_bboxes += num_of_bb_per_slp[i];
		counter--;
		if(counter == 0) {
			setScanLineOffs(ii_width,
					        load_balanced_bbox_offs,
							sl_map_list.at(cluster_idx),
							TOP_LEFT_CORNER,
							TOP_RIGHT_CORNER,
							num_of_bboxes);
			setScanLineOffs(ii_width,
					        load_balanced_bbox_offs,
							sl_map_list.at(cluster_idx),
							BOTTOM_LEFT_CORNER,
							BOTTOM_RIGHT_CORNER,
							num_of_bboxes);
			cum_sum_of_bb_per_cl[cluster_idx + 1] = num_of_bboxes + cum_sum_of_bb_per_cl[cluster_idx];
			//Set/Reset Fields for Next Cluster...
			load_balanced_bbox_offs += NUM_OF_BBOX_ATTRS_ON_GPU * num_of_bboxes;
			num_of_bboxes            = 0;
			counter                  = cluster_size;
			cluster_idx++;
		}//End of if-Block...
	}//End of Outer-for-Loop...
	
	if(counter > 0 && counter < cluster_size) {
		setScanLineOffs(ii_width, 
					    load_balanced_bbox_offs, 
						sl_map_list.at(cluster_idx), 
						TOP_LEFT_CORNER, 
						TOP_RIGHT_CORNER, 
						num_of_bboxes);
		setScanLineOffs(ii_width, 
					    load_balanced_bbox_offs, 
						sl_map_list.at(cluster_idx), 
						BOTTOM_LEFT_CORNER, 
						BOTTOM_RIGHT_CORNER, 
						num_of_bboxes);
		cum_sum_of_bb_per_cl[cluster_idx + 1] = num_of_bboxes + cum_sum_of_bb_per_cl[cluster_idx];
	}//End of-if-Block...
}

__host__
void findScanLines(int ii_width,
                   int *load_balanced_bbox_offs,
				   int total_num_of_slp,
				   std::vector<int>& num_of_bb_per_slp,
				   std::vector<SCAN_LINE_SET*>& scan_lines) {

	int i;
	SCAN_LINE_SET* temp;
	for(i = 0; i<total_num_of_slp; i++) {
		temp = new SCAN_LINE_SET();
		temp->insert(load_balanced_bbox_offs[TOP_LEFT_CORNER] / ii_width);
		temp->insert(load_balanced_bbox_offs[BOTTOM_LEFT_CORNER] / ii_width);
		scan_lines.push_back(temp);
		load_balanced_bbox_offs += NUM_OF_BBOX_ATTRS_ON_GPU * num_of_bb_per_slp[i];
	}//End of Outermost-for-Loop...
}

__host__
int findMaxNumSL(std::vector<SCAN_LINE_SET*>& scan_lines, 
                 int cluster_size,
				 int total_num_of_slp) {

	int i;
	SCAN_LINE_SET temp;
	
	int max_num_of_sl = 0;
	int counter       = cluster_size;
	for(i = 0; i<total_num_of_slp; i++) {
		temp.insert(*scan_lines[i]->begin());
		temp.insert(*(++scan_lines[i]->begin()));
		counter--;
		if(counter == 0) {
			if(temp.size() > max_num_of_sl) 
				max_num_of_sl = temp.size();
			counter = cluster_size;
			temp.clear();
		}
	}

	if(counter > 0 && counter < cluster_size)
		if(temp.size() > max_num_of_sl) 
			max_num_of_sl = temp.size();

	return max_num_of_sl;
}

__host__
void createClusters(int ii_width,
					TOTAL_RECALL_COMP_STR *trc,
					int *load_balanced_bbox_offs,
					int total_num_of_slp,
					std::vector<int>& num_of_bb_per_slp) {

	std::vector<SCAN_LINE_SET*> scan_lines;
	std::vector<SL_MAP*> sl_map_list;
	SL_MAP::iterator sl_map_it;
	SL_MAP* temp_sl_map;

	std::vector<int>::iterator it, it2;
	size_t required_sm_size;
	host_vector<int> temp_arr;
	int i, j;
	int max_num_of_sl;
	int max_active_warps;
	int cluster_size;
	int num_of_tpb;
	int required_warps;
	int num_of_warps;
	int temp_size;

	cudaDeviceProp* prop = getDeviceProps(0);

	//**********************************************************************
	//*******************Find Optimum # of BPG & TPB...*********************
	//**********************************************************************
	//Constraint #1)Maximum # of Threads per Block...
	num_of_tpb = prop->regsPerBlock / TH_REG_COUNT_FOR_PV_COMP;//Max TPB...
	//Here It Refers to the # of Threads to Hide Register-Latency...
	temp_size = NUM_OF_CYCLES_FOR_RL * getNumOfCUDACoresPerSM(0);
	temp_size = temp_size <= MIN_NUM_OF_TPB ? MIN_NUM_OF_TPB : temp_size;
	temp_size = temp_size >= prop->maxThreadsPerBlock ? prop->maxThreadsPerBlock : temp_size;
	//Try to Utilize the Device...
	if(num_of_tpb < MIN_NUM_OF_TPB)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_NOT_SUFFICIENT_GPU_RES),
		                        "Computing Patch Variance(Register Constraint)"),
						 fast_detection,
						 ER_FASTDET_NOT_SUFFICIENT_GPU_RES);
	else if(trc->num_of_bboxes < MIN_NUM_OF_TPB)
		num_of_tpb = MIN_NUM_OF_TPB;
	else if(num_of_tpb >= temp_size)
		num_of_tpb = temp_size;
	else 
		num_of_tpb = (int)floorf(num_of_tpb / (float)MIN_NUM_OF_TPB) * MIN_NUM_OF_TPB;
	//Constraint #2)Maximum # of Resident Threads per Multiprocessor...
	if(MAX_NUM_OF_BLOCKS_PER_SM * num_of_tpb > prop->maxThreadsPerMultiProcessor) {
		temp_size                = (int)floorf(prop->maxThreadsPerMultiProcessor / 
			                                   (float)MAX_NUM_OF_BLOCKS_PER_SM);
		num_of_tpb = temp_size - (temp_size % 32);
	}
	//Constraint #3)Shared Memory Size for Scan-Line Pairs(for Both II, and II2)...
	required_sm_size = 2 * sizeof(Npp32s) * ii_width;//*2 For Bottom and Top Rows of a Single slp...
	if(prop->sharedMemPerBlock < required_sm_size)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_NOT_SUFFICIENT_GPU_RES),
		                        "Computing Patch Variance(Shared Memory Constraint)"),
						 fast_detection,
						 ER_FASTDET_NOT_SUFFICIENT_GPU_RES);
	findScanLines(ii_width,
				  load_balanced_bbox_offs,
				  total_num_of_slp,
				  num_of_bb_per_slp,
				  scan_lines);
	cluster_size = 1;
	while(cluster_size < total_num_of_slp) {
		max_num_of_sl = findMaxNumSL(scan_lines, 
			                         cluster_size + 1, 
									 total_num_of_slp);
		if(prop->sharedMemPerBlock < (sizeof(Npp32s) * max_num_of_sl * ii_width))
			break;
		required_sm_size = sizeof(Npp32s) * max_num_of_sl * ii_width;
		cluster_size++;
	}//End of while-Loop...
	//**********************************************************************
	//*****What if Occupancy is Too Low With Respect to Those Params...*****
	//**********************************************************************
	temp_size = prop->maxThreadsPerMultiProcessor / 32; //Num of Active Warps an SM May Support...
	do {
		max_active_warps = (num_of_tpb / 32) * 
			               (int)floorf(prop->sharedMemPerBlock / (float)required_sm_size);
		if(cluster_size < 2 || (max_active_warps / (float)temp_size) >= MIN_OCCUPANCY_PERC) 
			break;
		cluster_size--;
	    max_num_of_sl    = findMaxNumSL(scan_lines, 
										cluster_size, 
										total_num_of_slp);
		required_sm_size = sizeof(Npp32s) * max_num_of_sl * ii_width;
	}while(1);
	//Find # of Clusters...
	trc->num_of_clusters = (int)ceilf(total_num_of_slp / (float)cluster_size);
	trc->pv_comp_sm_size = required_sm_size;
	//Find # of Blocks Per Invocation...
	temp_size = MAX_NUM_OF_BLOCKS_PER_SM * prop->multiProcessorCount;
	if(trc->num_of_clusters < temp_size)
		trc->max_num_of_bpg = trc->num_of_clusters;
	else
		trc->max_num_of_bpg = temp_size;
	//Find # of Async Invocations...
	trc->num_of_async_inv = (int)ceilf(trc->num_of_clusters / (float)trc->max_num_of_bpg);
	//Find x and y Dimensions of Each Block...
	num_of_warps   = num_of_tpb / 32;
	required_warps = (int)ceilf(ii_width / (float)WARP_SIZE);
	if(num_of_warps > required_warps) {
		while((required_warps - 1) >= 1) {
			required_warps--;
			if(num_of_tpb % (WARP_SIZE * required_warps) == 0) {
				num_of_warps = required_warps;
				break;
			}
		}//End of while-Loop...
	}//End of if-Block...
	trc->pv_comp_block_dim.x = WARP_SIZE * num_of_warps;
	trc->pv_comp_block_dim.y = num_of_tpb / trc->pv_comp_block_dim.x;
	trc->pv_comp_block_dim.z = 1;
	//**********************************************************************
	//*******************Find Scan-Lines' SM Offsets...*********************
	//**********************************************************************
	if(cluster_size > 1)
		mergeScanLines(cluster_size, scan_lines);
	findSMIndicesOfSLs(sl_map_list,
		               scan_lines,
					   cluster_size);
	//**********************************************************************
	//***Now We are Ready to Define Clusters and All Related Parameters*****
	//************within TLD_OBJECT's fast_detection Structure...***********
	//**********************************************************************
	temp_size = sl_map_list.size();
	assert(temp_size == fast_det_str->num_of_clusters);//They Must be Equal to Each Other...
	//Find Cumulative Sum of # of BBOXes for All Clusters In the Meantime...
	temp_arr.resize(trc->num_of_clusters + 1);
	setSMIndicesofBBs(load_balanced_bbox_offs, 
		              sl_map_list,
					  cluster_size,
					  total_num_of_slp,
					  ii_width,
					  num_of_bb_per_slp,
					  temp_arr);
	trc->dev_cum_sum_of_bb_per_cluster = new thrust::device_vector<int>();
	trc->dev_cum_sum_of_bb_per_cluster->resize(trc->num_of_clusters + 1);
	trc->cum_sum_of_bb_per_cluster = new thrust::host_vector<int>();
	trc->cum_sum_of_bb_per_cluster->resize(trc->num_of_clusters + 1);
	thrust::copy(temp_arr.begin(), 
		         temp_arr.end(), 
				 trc->dev_cum_sum_of_bb_per_cluster->begin());
	thrust::copy(temp_arr.begin(), 
		         temp_arr.end(), 
				 trc->cum_sum_of_bb_per_cluster->begin());
	temp_arr.clear();
	//Find Cumulative Sum of # of Scan-Lines...
	temp_arr.resize(trc->num_of_clusters + 1);
	temp_arr[0]   = 0;
	max_num_of_sl = 0;
	for(i = 0; i<temp_size; i++) {
		temp_arr[i + 1] = sl_map_list.at(i)->size() + temp_arr[i];
		if(sl_map_list.at(i)->size() > max_num_of_sl) 
			max_num_of_sl = sl_map_list.at(i)->size();
	}//End of for-Loop...
	trc->dev_cum_sum_of_scan_lines = new thrust::device_vector<int>();
	trc->dev_cum_sum_of_scan_lines->resize(trc->num_of_clusters + 1);
	thrust::copy(temp_arr.begin(), 
		         temp_arr.end(), 
				 trc->dev_cum_sum_of_scan_lines->begin());
	temp_size = *(temp_arr.end() - 1);//Total # of Scan-Lines...
	temp_arr.clear();
	//Set Shared Memory Indices of Scan-Lines for Each Cluster...
	temp_arr.resize(temp_size);
	temp_size = sl_map_list.size();
	j         = 0;
	for(i = 0; i<temp_size; i++) {
		temp_sl_map = sl_map_list.at(i);
		for(sl_map_it = temp_sl_map->begin();
			sl_map_it != temp_sl_map->end();
			++sl_map_it) {
			temp_arr[sl_map_it->second + j] = sl_map_it->first;
		}//End of Inner-for-Loop...
		j += temp_sl_map->size();
	}//End of for-Loop...
	trc->dev_scan_line_rows = new thrust::device_vector<int>();
	trc->dev_scan_line_rows->resize(temp_arr.size());
	thrust::copy(temp_arr.begin(), 
		         temp_arr.end(), 
				 trc->dev_scan_line_rows->begin());
	//**********************************************************************
	//*********************Free Allocated Resources...**********************
	//**********************************************************************
	temp_size = sl_map_list.size();
	for(i = 0; i<temp_size; i++)
		delete sl_map_list.at(i);
	sl_map_list.clear();
	temp_size = scan_lines.size();
	for(i = 0; i<temp_size; i++)
		delete scan_lines.at(i);
	scan_lines.clear();
}

//********************************************************************************
//**********Copying Load Balanced Chunks into Destination Location...*************
//********************************************************************************
__host__
void copyBBOffs(int *bbox_offs,
                int st_bb_idx,
				int slp_bb_size,
				size_t copy_size,
				bool is_reverse,
				int *start_offs, 
				int *bb_idx) {

	int i;

	if(is_reverse) {
		bbox_offs -= NUM_OF_BBOX_ATTRS_ON_CPU * slp_bb_size;
		st_bb_idx -= slp_bb_size;
	}
	for(i = 0; i<slp_bb_size; i++) {
		memcpy((void*)start_offs,
			   (void*)bbox_offs,
			   copy_size);
		bb_idx[i]   = st_bb_idx++;
		bbox_offs  += NUM_OF_BBOX_ATTRS_ON_CPU;
		start_offs += NUM_OF_BBOX_ATTRS_ON_GPU;
	}
}

__host__
void balanceLoad(int *bbox_offs,
				 int *num_of_slp_per_sl,
				 int *num_of_lrb_per_sl,
				 int total_num_of_slp,
				 int num_of_scales,
				 int num_of_bboxes, 
				 int *load_balanced_bbox_offs,
				 int *bb_idx,
				 std::vector<int>& num_of_bb_per_slp) {

	int i;
	size_t copy_size;
	int half_size;
	int front_num_of_slp, rear_num_of_slp;
	int front_sl, rear_sl;
	int front_bb_idx, rear_bb_idx;
	int *end_ptr;

	front_sl         = 0;
	rear_sl          = num_of_scales - 1;
	front_num_of_slp = num_of_slp_per_sl[front_sl];
	rear_num_of_slp  = num_of_slp_per_sl[rear_sl];
	end_ptr          = bbox_offs + num_of_bboxes * NUM_OF_BBOX_ATTRS_ON_CPU;
	copy_size        = sizeof(int) * NUM_OF_BBOX_ATTRS_ON_GPU;
	front_bb_idx     = 0;
	rear_bb_idx      = num_of_bboxes;
	half_size        = total_num_of_slp / 2;
	for(i = 0; i<half_size; i++) {
		//Copy Scan-Line Pair At the Beginning...
		copyBBOffs(bbox_offs,
				   front_bb_idx,
				   num_of_lrb_per_sl[front_sl],
				   copy_size,
				   false,
				   load_balanced_bbox_offs, 
				   bb_idx);
		//Copy Scan-Line Pair At the End...
		load_balanced_bbox_offs += num_of_lrb_per_sl[front_sl] * NUM_OF_BBOX_ATTRS_ON_GPU;
		bb_idx                  += num_of_lrb_per_sl[front_sl];
		num_of_bb_per_slp.push_back(num_of_lrb_per_sl[front_sl]);
		copyBBOffs(end_ptr,
				   rear_bb_idx,
				   num_of_lrb_per_sl[rear_sl],
				   copy_size,
				   true,
				   load_balanced_bbox_offs, 
				   bb_idx);
		load_balanced_bbox_offs += num_of_lrb_per_sl[rear_sl] * NUM_OF_BBOX_ATTRS_ON_GPU;
		bb_idx                  += num_of_lrb_per_sl[rear_sl];
		num_of_bb_per_slp.push_back(num_of_lrb_per_sl[rear_sl]);
		//Update Front...
		front_num_of_slp--;
		front_bb_idx += num_of_lrb_per_sl[front_sl];
		bbox_offs    += num_of_lrb_per_sl[front_sl] * NUM_OF_BBOX_ATTRS_ON_CPU;
		if(front_num_of_slp < 1) {
			front_sl++;
			front_num_of_slp = num_of_slp_per_sl[front_sl];
		}
		//Update Rear...
		rear_num_of_slp--;
		rear_bb_idx -= num_of_lrb_per_sl[rear_sl];
		end_ptr     -= num_of_lrb_per_sl[rear_sl] * NUM_OF_BBOX_ATTRS_ON_CPU;
		if(rear_num_of_slp < 1) {
			rear_sl--;
			rear_num_of_slp = num_of_slp_per_sl[rear_sl];
		}
	}//End of for-Loop...
	//If it Is Odd Numbered...
	if((total_num_of_slp & 1) > 0) {
		copyBBOffs(bbox_offs,
				   front_bb_idx,
				   num_of_lrb_per_sl[front_sl],
				   copy_size,
				   false,
				   load_balanced_bbox_offs, 
				   bb_idx);
		num_of_bb_per_slp.push_back(num_of_lrb_per_sl[front_sl]);
	}//End of if-Block...
}

__host__
void FastDetection::findKLPForBBOXProcessing(int num_of_bboxes,
                                             int &grid_dim,
											 int &block_dim) {

	int num_of_req_bl;
	int num_of_sm;
	cudaDeviceProp *prop;

	//Minimal KLP....
	if(num_of_bboxes < WARP_SIZE) {
		grid_dim  = 1;
		block_dim = WARP_SIZE;
		return;
	}
	prop          = getDeviceProps(0);
	num_of_sm     = prop->multiProcessorCount;
	num_of_req_bl = (int)ceilf(num_of_bboxes / (float)prop->maxThreadsPerBlock);
	//Maximal KLP...
	if(num_of_req_bl >= num_of_sm * MAX_NUM_OF_BLOCKS_PER_SM) {
		grid_dim  = num_of_sm * MAX_NUM_OF_BLOCKS_PER_SM;
		block_dim = prop->maxThreadsPerBlock;
		return;
	}
	grid_dim  = num_of_req_bl;
	block_dim = prop->maxThreadsPerBlock;
	if(num_of_req_bl < num_of_sm) {
		do {
			block_dim -= WARP_SIZE;
			grid_dim   = (int)ceilf(num_of_bboxes / (float)block_dim);
		}while(grid_dim < num_of_sm && block_dim > 2 * WARP_SIZE);
	}
}

//********************************************************************************
//************Initializing TLDObject's Fast-Detection Structure*******************
//********************************************************************************
__host__
void FastDetection::initializeTLDObject(FAST_DETECTION_STR *fast_det,
		                                int num_of_sl,
										int num_of_trees,
										int num_of_features,
		                                int *num_of_bboxes_per_sl,
										int *num_of_lrb_per_sl,
										int *bbox_offs,
										int *forest_offs) {

	cudaError cuda_status;
	int i;
	int total_num_of_cl;
	int exec_off;
	int total_num_of_slp;
	std::vector<int> num_of_bb_per_slp;
	int chunk_size;
	int *load_balanced_bbox_offs = NULL;
	int *num_of_slp_per_sl       = NULL;
	int * abs_tl_corner          = NULL;
	TOTAL_RECALL_COMP_STR *trc   = NULL;

	fast_det->trc = (TOTAL_RECALL_COMP_STR*)malloc(sizeof(TOTAL_RECALL_COMP_STR));
	trc           = fast_det->trc;
	trc->num_of_bboxes = 0;
	for(i = 0; i<num_of_sl; i++)
		trc->num_of_bboxes += num_of_bboxes_per_sl[i];
	trc->num_of_sl        = num_of_sl;
	trc->num_of_trees     = num_of_trees;
	trc->num_of_features  = num_of_features;
	trc->num_indices      = (int)pow(2.0f, num_of_features);
	trc->rfi_comp_sm_size = num_of_sl * num_of_trees * num_of_features * 2 * sizeof(int);
	//*******************************************************************************
	//*********Allocate Necessary Fields for Total Recall Computation on GPU*********
	//*******************************************************************************
	//*********Device Side Allocation...
	cuda_status = cudaMalloc((void**)(&trc->dev_bb_var), 
		                     sizeof(float) * trc->num_of_bboxes);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR),
								"BB Variances on GPU(Allocating)",
								cudaGetErrorString(cuda_status)),
						 fast_detection,
						 ER_FASTDET_CUDA_MEM_ERROR);

	cuda_status = cudaMalloc((void**)(&trc->dev_pv_status), 
		                     sizeof(int) * trc->num_of_bboxes);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR),
								"PV Status on GPU(Allocating)",
								cudaGetErrorString(cuda_status)),
						 fast_detection,
						 ER_FASTDET_CUDA_MEM_ERROR);

	cuda_status = cudaMalloc((void**)(&trc->dev_bb_lsh), 
		                     sizeof(int) * trc->num_of_bboxes);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR),
								"BBOX Backward Shift Amount on GPU(Allocating)",
								cudaGetErrorString(cuda_status)),
						 fast_detection,
						 ER_FASTDET_CUDA_MEM_ERROR);

	cuda_status = cudaMalloc((void**)(&trc->dev_bb_rsh), 
		                     sizeof(int) * trc->num_of_bboxes);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR),
								"BBOX Forward Shift Amount on GPU(Allocating)",
								cudaGetErrorString(cuda_status)),
						 fast_detection,
						 ER_FASTDET_CUDA_MEM_ERROR);

	cuda_status = cudaMalloc((void**)(&trc->dev_rfis), 
							 sizeof(int) * trc->num_of_bboxes * num_of_trees);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR),
								"Conf Indices on GPU(Allocating)",
								cudaGetErrorString(cuda_status)),
						 fast_detection,
						 ER_FASTDET_CUDA_MEM_ERROR);

	cuda_status = cudaMalloc((void**)(&trc->dev_abs_tl_corner), 
		                     sizeof(int) * trc->num_of_bboxes);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR),
								"Absolute TOP-LEFT Corner Values on GPU(Allocating)",
								cudaGetErrorString(cuda_status)),
						 fast_detection,
						 ER_FASTDET_CUDA_MEM_ERROR);
	//*********Host Side Allocation...
	cuda_status = cudaHostAlloc((void**)&trc->bb_var, 
		                        sizeof(float) * trc->num_of_bboxes, 
								cudaHostAllocPortable);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR),
								"BB Variances on CPU(Allocating)",
								cudaGetErrorString(cuda_status)),
						 fast_detection,
						 ER_FASTDET_CUDA_MEM_ERROR);
	cuda_status = cudaHostAlloc((void**)&trc->bb_idx, 
		                        sizeof(int) * trc->num_of_bboxes, 
								cudaHostAllocPortable);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR),
								"BB Indices on CPU(Allocating)",
								cudaGetErrorString(cuda_status)),
						 fast_detection,
						 ER_FASTDET_CUDA_MEM_ERROR);

	cuda_status = cudaHostAlloc((void**)&trc->gpu_rfis, 
								sizeof(int) * trc->num_of_bboxes * num_of_trees, 
								cudaHostAllocPortable);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR),
								"Conf Indices on CPU(Allocating)",
								cudaGetErrorString(cuda_status)),
						 fast_detection,
						 ER_FASTDET_CUDA_MEM_ERROR);
	
	cuda_status = cudaHostAlloc((void**)&trc->bb_rsh_amount, 
								sizeof(int) * trc->num_of_bboxes, 
								cudaHostAllocPortable);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR),
								"BB Shift Amount on CPU(Allocating)",
								cudaGetErrorString(cuda_status)),
						 fast_detection,
						 ER_FASTDET_CUDA_MEM_ERROR);
	//**********************************************************************
	//***********************Balancing the Load...**************************
	//**********************************************************************
	load_balanced_bbox_offs = (int*)malloc(sizeof(int) * 
		                                   NUM_OF_BBOX_ATTRS_ON_GPU * 
										   trc->num_of_bboxes);
	num_of_slp_per_sl       = (int*)malloc(sizeof(int) * num_of_sl);
	total_num_of_slp        = 0;
	for(i = 0; i<num_of_sl; i++) {
		num_of_slp_per_sl[i] = num_of_bboxes_per_sl[i] / num_of_lrb_per_sl[i];
		total_num_of_slp    += num_of_slp_per_sl[i];
	}//End of for-Loop...
	balanceLoad(bbox_offs,
				num_of_slp_per_sl,
				num_of_lrb_per_sl,
				total_num_of_slp,
				num_of_sl,
				trc->num_of_bboxes,
				load_balanced_bbox_offs,
				trc->bb_idx,
				num_of_bb_per_slp);
	//**********************************************************************
	//******************Right Before Creating Clusters,*********************
	//*********************Move Abs TL_CORNER Info;*************************
	//******************Because They Will Be Replaced With *****************
	//*******************Their SM Address Equivalents...********************
	//**********************************************************************
	abs_tl_corner = (int*)malloc(sizeof(int) * trc->num_of_bboxes);
	for(i = 0; i<trc->num_of_bboxes; i++)
		abs_tl_corner[i] = load_balanced_bbox_offs[i * NUM_OF_BBOX_ATTRS_ON_GPU + TOP_LEFT_CORNER];
	cudaMemcpy((void*)trc->dev_abs_tl_corner,
			   (void*)abs_tl_corner,
			   sizeof(int) * trc->num_of_bboxes,
			   cudaMemcpyHostToDevice);
	//**********************************************************************
	//**************************Create Clusters...**************************
	//**********************************************************************
	createClusters(_mem_module->getSROI()->width,
				   fast_det->trc,
				   load_balanced_bbox_offs,
				   total_num_of_slp,
				   num_of_bb_per_slp);
	//*******************************************************************************
	//*********Allocate Necessary Fields for Total Recall Computation on GPU*********
	//*******(Fields That Might be Only Allocated After Cluster Creation Time)*******
	//*******************************************************************************
	//This is Avg Chunk Size(A Thread May Not Process More Than This # of BBOXes)...
	//Check "Equivalences" Part of http://en.wikipedia.org/wiki/Floor_and_ceiling_functions (Why +num_of_async_inv)
	chunk_size = (int)ceilf(trc->num_of_bboxes / 
		                              (float)_num_of_cpu_threads) + trc->num_of_async_inv;
	trc->pv_streams        = new cudaStream_t[trc->num_of_async_inv];
	trc->pv_events         = new cudaEvent_t[trc->num_of_async_inv];
	trc->crfi_streams      = new cudaStream_t[trc->num_of_async_inv];
	trc->crfi_events       = new cudaEvent_t[trc->num_of_async_inv];
	trc->exec_params       = new int[trc->num_of_async_inv * TRC_EXEC_PARAMS_COUNT];
	trc->sh_temp_buf_size  = new size_t[trc->num_of_async_inv];
	trc->dev_sh_am_is_buf  = new void*[trc->num_of_async_inv];
	trc->st_comp_grid_dim  = new int[trc->num_of_async_inv];
	trc->st_comp_block_dim = new int[trc->num_of_async_inv];
	trc->tpri_exec_params  = new int*[_num_of_cpu_threads];
	trc->tpri_out_conf     = new float*[_num_of_cpu_threads];
	trc->ready_for_copy    = new int[_num_of_cpu_threads];
	for(i = 0; i<_num_of_cpu_threads; i++) {
		trc->tpri_exec_params[i]    = new int[trc->num_of_async_inv * TPRI_TRC_EXEC_PARAMS_COUNT];
		trc->tpri_exec_params[i][0] = 0;//Always Starts at 0th Position...
		trc->tpri_out_conf[i]       = new float[chunk_size];
	}
	//Create CUDA Streams & Events for Total Recall Computation...
	for(i = 0; i<trc->num_of_async_inv; i++) {
		cudaEventCreateWithFlags(&(trc->pv_events[i]), 
					             cudaEventDisableTiming);
		cudaStreamCreate(&(trc->pv_streams[i]));
		cudaEventCreateWithFlags(&(trc->crfi_events[i]), 
					             cudaEventDisableTiming);
		cudaStreamCreate(&(trc->crfi_streams[i]));
	}
	//Initialize Async Execution Parameters for Total Recall Computation...
	for(i = 0; i<trc->num_of_async_inv; i++) {
		exec_off                            = i * TRC_EXEC_PARAMS_COUNT;
		total_num_of_cl                     = trc->num_of_clusters - i * trc->max_num_of_bpg;
		total_num_of_cl                     = total_num_of_cl >  trc->max_num_of_bpg ? 
											      trc->max_num_of_bpg : total_num_of_cl;
		trc->exec_params[exec_off + TRC_EXEC_PV_INV_OFF]   = i * trc->max_num_of_bpg;
		trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF] = 
			(*trc->cum_sum_of_bb_per_cluster)[trc->exec_params[exec_off + TRC_EXEC_PV_INV_OFF]];
		trc->exec_params[exec_off + TRC_EXEC_PV_TOTAL_NUM_OF_BB] = 
			(*trc->cum_sum_of_bb_per_cluster)[trc->exec_params[exec_off + TRC_EXEC_PV_INV_OFF] + total_num_of_cl] - 
			trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF];
		trc->exec_params[exec_off + TRC_EXEC_PV_TOTAL_NUM_OF_CL] = total_num_of_cl;
		//Find Kernel Launch Parameters for BBOX Stream Compaction...
		findKLPForBBOXProcessing(trc->exec_params[exec_off + TRC_EXEC_PV_TOTAL_NUM_OF_BB],
								 trc->st_comp_grid_dim[i],
								 trc->st_comp_block_dim[i]);
		//Find Buffer Size for Inclusive Sum Operation Per Async Invocation...
		trc->dev_sh_am_is_buf[i] = NULL;
		DeviceScan::InclusiveSum(trc->dev_sh_am_is_buf[i],
								 trc->sh_temp_buf_size[i],
								 trc->dev_pv_status + trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF],
								 trc->dev_bb_lsh + trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF],
								 trc->exec_params[exec_off + TRC_EXEC_PV_TOTAL_NUM_OF_BB]);
		cuda_status = cudaMalloc(&trc->dev_sh_am_is_buf[i],
								 trc->sh_temp_buf_size[i]);
		if(cuda_status != cudaSuccess)
			throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR),
									"Temp Buffer for Shift Amount IS Op on GPU(Allocating)",
									cudaGetErrorString(cuda_status)),
							 fast_detection,
							 ER_FASTDET_CUDA_MEM_ERROR);
	}//End of for-Loop...
	//**********************************************************************
	//**********************Moving BBOX Offs to GPU...**********************
	//**********************************************************************
	cuda_status = cudaMalloc((void**)&trc->dev_bbox_offs, 
		                     sizeof(int) * 
							 NUM_OF_BBOX_ATTRS_ON_GPU * 
							 trc->num_of_bboxes);
	if(cuda_status != cudaSuccess) {
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR), 
								"BBOX Offs for PV Comp.",
								cudaGetErrorString(cuda_status)), 
						 fast_detection, 
						 ER_FASTDET_CUDA_MEM_ERROR);
	} else 
		_mem_module->moveBBOXOffsToGPU(load_balanced_bbox_offs,
									   trc->dev_bbox_offs,
									   trc->num_of_bboxes);
	//**********************************************************************
	//********************Moving Forests Offs to GPU...*********************
	//**********************************************************************
	cuda_status = cudaMalloc((void**)&trc->dev_forests_offs, 
							 sizeof(int) * 2 * num_of_features * num_of_trees * num_of_sl);
	if(cuda_status != cudaSuccess) {
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR), 
								"Forest Offs",
								cudaGetErrorString(cuda_status)), 
						 fast_detection, 
						 ER_FASTDET_CUDA_MEM_ERROR);
	} else 
		_mem_module->moveForestsOffsToGPU(forest_offs,
										  trc->dev_forests_offs,
										  num_of_sl,
										  num_of_trees,
										  num_of_features);
	//Free All Allocated Resources...
	free((void*)load_balanced_bbox_offs);
	free((void*)num_of_slp_per_sl);
	free((void*)abs_tl_corner);
}

//********************************************************************************
//**************Finalizing TLDObject's Fast-Detection Structure*******************
//********************************************************************************
__host__
void FastDetection::finalizeTLDObject(FAST_DETECTION_STR *fast_det) {

	int i;
	TOTAL_RECALL_COMP_STR * trc = fast_det->trc;
	//************************************
	//***********Device Side...***********
	//************************************
	if(trc->dev_cum_sum_of_bb_per_cluster != NULL) 
		delete trc->dev_cum_sum_of_bb_per_cluster;
	if(trc->dev_cum_sum_of_scan_lines != NULL) 
		delete trc->dev_cum_sum_of_scan_lines;
	if(trc->dev_scan_line_rows != NULL) 
		delete trc->dev_scan_line_rows;
	if(trc->dev_bbox_offs != NULL)
		cudaFree((void*)trc->dev_bbox_offs);
	if(trc->dev_bb_var != NULL) 
		cudaFree((void*)trc->dev_bb_var);
	if(trc->dev_forests_offs != NULL)
		cudaFree((void*)trc->dev_forests_offs);
	if(trc->dev_pv_status != NULL) 
		cudaFree((void*)trc->dev_pv_status);
	if(trc->dev_bb_lsh != NULL) 
		cudaFree((void*)trc->dev_bb_lsh);
	if(trc->dev_bb_rsh != NULL) 
		cudaFree((void*)trc->dev_bb_rsh);
	if(trc->dev_rfis != NULL) 
		cudaFree((void*)trc->dev_rfis);
	if(trc->dev_abs_tl_corner != NULL) 
		cudaFree(trc->dev_abs_tl_corner);
	//Destroy all Resources Associated with Async Invocation #...
	if(trc->dev_sh_am_is_buf != nullptr) {
		for(i = 0; i<trc->num_of_async_inv; i++)
			cudaFree((void*)trc->dev_sh_am_is_buf[i]);
		delete[] trc->dev_sh_am_is_buf;
	}
	//************************************
	//************Host Side...************
	//************************************
	if(trc->cum_sum_of_bb_per_cluster != NULL) 
		delete trc->cum_sum_of_bb_per_cluster;
	if(trc->bb_var != NULL)
		cudaFreeHost((void*)trc->bb_var);
	if(trc->bb_idx != NULL)
		cudaFreeHost((void*)trc->bb_idx);
	if(trc->gpu_rfis != NULL)
		cudaFreeHost((void*)trc->gpu_rfis);
	if(trc->bb_rsh_amount != NULL)
		cudaFreeHost((void*)trc->bb_rsh_amount);
	//Destroy all Resources Associated with Async Invocation #...
	for(i = 0; i<trc->num_of_async_inv; i++) {
		cudaStreamDestroy(trc->pv_streams[i]);
		cudaEventDestroy(trc->pv_events[i]);
		cudaStreamDestroy(trc->crfi_streams[i]);
		cudaEventDestroy(trc->crfi_events[i]);
	}
	if(trc->pv_streams != nullptr)
		delete[] trc->pv_streams;
	if(trc->pv_events != nullptr)
		delete[] trc->pv_events;
	if(trc->crfi_streams != nullptr)
		delete[] trc->crfi_streams;
	if(trc->crfi_events != nullptr)
		delete[] trc->crfi_events;
	if(trc->sh_temp_buf_size != nullptr)
		delete[] trc->sh_temp_buf_size;
	if(trc->st_comp_grid_dim != nullptr)
		delete[] trc->st_comp_grid_dim;
	if(trc->st_comp_block_dim != nullptr)
		delete[] trc->st_comp_block_dim;
	if(trc->exec_params != nullptr)
		delete[] trc->exec_params;
	if(trc->tpri_out_conf != nullptr) {
		for(i = 0; i<_num_of_cpu_threads; i++)
			delete[] trc->tpri_out_conf[i];
		delete[] trc->tpri_out_conf;
	}
	if(trc->tpri_exec_params != nullptr) {
		for(i = 0; i<_num_of_cpu_threads; i++)
			delete[] trc->tpri_exec_params[i];
		delete[] trc->tpri_exec_params;
	}
	if(trc->ready_for_copy != nullptr)
		delete[] trc->ready_for_copy;
}

//********************************************************************************
//********************************Computing IIs...********************************
//********************************************************************************
__host__
void FastDetection::computeIIs(bool is_move_to_host) {

	NppStatus npp_status;
	//Now Compute Them on GPU By Using NPPI...
	npp_status = nppiSqrIntegral_8u32s_C1R(_mem_module->getDevCurFrame(),
		                                   sizeof(Npp8u) * _mem_module->getDROI()->width,
										   _mem_module->getDevII(),
										   sizeof(Npp32s) * (_mem_module->getDROI()->width + 1),
										   _mem_module->getDevII2(),
										   sizeof(Npp32s) * (_mem_module->getDROI()->width + 1),
										   *(_mem_module->getSROI()),
										   (Npp32s)0,
										   (Npp32s)0);
	if(npp_status != NPP_SUCCESS)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_SQR_INT_IMG_COMP), 
								"computeIIs", 
								(int)(npp_status)), 
						 fast_detection,
						 ER_FASTDET_SQR_INT_IMG_COMP);
	if(is_move_to_host)
		_mem_module->moveIIsToHost();
}

//********************************************************************************
//***************************Computing Total Recall...****************************
//********************************************************************************

//********************************************************************************
//****************************Part#1)PV Computation...****************************
//********************************************************************************
__global__ void computePVOnGPU(const int* __restrict__ bbox_offs,
							   Npp32s **iis,
							   const int* __restrict__ cum_sum_of_scan_lines,
							   const int* __restrict__ scan_line_rows,
							   const int* __restrict__ cum_sum_of_bb_per_cluster,
							   float min_var,
							   float* __restrict__ bb_var,
							   int* __restrict__ pv_status) {
	//Dynamically Allocated Shared Memory. That is the Reason for Why it is Declared as "extern"...
	extern __shared__ Npp32s scan_lines[];
	int common;
	int i, j, k;
	int sl_st_idx, sl_size;
	int bb_st_idx, bb_size;
	float mx;

	sl_st_idx = cum_sum_of_scan_lines[blockIdx.x];
	bb_st_idx = cum_sum_of_bb_per_cluster[blockIdx.x];
	sl_size   = cum_sum_of_scan_lines[blockIdx.x + 1] - sl_st_idx;
	bb_size   = cum_sum_of_bb_per_cluster[blockIdx.x + 1] - bb_st_idx;

	//**************************************************************************
	//**************************Part#1)PV Computation***************************
	//**************************************************************************
	for(k = 0; k<2; k++) {
		//**************************************************************************
		//*****************Sub-Part#1)Load SLPs into Shared Memory******************
		//**************************************************************************
		for(i = threadIdx.y; i<sl_size; i += blockDim.y) {
			//Top Border is All Zero(Due to NPP's II Computation)... Add 1...
			common = scan_line_rows[sl_st_idx + i] + 1;//Here it Refers to Row ID of the Current Scan-Line to be Read...
			for(j = threadIdx.x; j<PV_INT_CONSTS[0]; j += blockDim.x)
				scan_lines[PV_INT_CONSTS[0] * i + j] = iis[k][common * PV_INT_CONSTS[1] + 1 + j];
		}//End of Outer-for-Loop...
		__syncthreads();
		//**************************************************************************
		//***********Sub-Part#2)Compute PV for Each BBOX in this Cluster************
		//**************************************************************************
		common = blockDim.x * blockDim.y;//Here it Refers to the Total Block Size...
		for(i = blockDim.x * threadIdx.y + threadIdx.x; i<bb_size; i += common) {
			mx = (scan_lines[bbox_offs[bb_st_idx + i + TLD_OBJ_INT_CONSTS[0] * BOTTOM_RIGHT_CORNER]] - 
				  scan_lines[bbox_offs[bb_st_idx + i + TLD_OBJ_INT_CONSTS[0] * TOP_RIGHT_CORNER]] - 
				  scan_lines[bbox_offs[bb_st_idx + i + TLD_OBJ_INT_CONSTS[0] * BOTTOM_LEFT_CORNER]] + 
				  scan_lines[bbox_offs[bb_st_idx + i + TLD_OBJ_INT_CONSTS[0] * TOP_LEFT_CORNER]]) / 
				  (float)bbox_offs[bb_st_idx + i + TLD_OBJ_INT_CONSTS[0] * AREA_OF_SCANNING_WINDOW];
			bb_var[bb_st_idx + i] = (k == 0 ? -(mx * mx) : bb_var[bb_st_idx + i] + mx);
		}//End of for-Loop...
		__syncthreads();
	}//End of Outermost-for-Loop...
	
	//**************************************************************************
	//***********************Part#2)Filtering Out BBOXes************************
	//**************************************************************************
	for(i = blockDim.x * threadIdx.y + threadIdx.x; i<bb_size; i += common)
		pv_status[bb_st_idx + i] = bb_var[bb_st_idx + i] >= min_var ? 0 : -1;
}

//********************************************************************************
//****************************Part #2)Finding RSH...******************************
//********************************************************************************
__global__ void findRightShifts(int num_of_bb,
								const int* __restrict__ pv_status,
								const int* __restrict__ bb_lsh,
								int* __restrict__ bb_rsh) {

	int i;
	int shift;
	int jmp_amount = gridDim.x * blockDim.x;
	for(i = blockDim.x * blockIdx.x + threadIdx.x; i<num_of_bb; i += jmp_amount) {
		if(pv_status[i] == 0) {
			shift             = bb_lsh[i];
			bb_rsh[i + shift] = -1 * shift;
		}
	}
}

//********************************************************************************
//************************Part #3)RF Index Computation...*************************
//********************************************************************************
__global__ void computeRFIndicesOnGPU(int start,
								      int end,
									  int num_of_bb,
									  const ubyte* __restrict__ blurred_image,//Access Pattern is Not Coalesced...
									  const int* __restrict__ bbox_offs,//Access Pattern is Not Coalesced...
									  const int* __restrict__ abs_tl_corner,//Access Pattern is Not Coalesced...
									  const int* __restrict__ forests_offs,//Coalesced Access Guaranteed...
								      const int* __restrict__ bb_sh_amount,//Coalesced Access Guaranteed...
									  int* __restrict__ conf_indices//Coalesced Access Guaranteed...
									  ) {

	extern __shared__ int features[];
	int i, j, k;
	int index;
	int bb_idx;
	int feature_ptr;
	int tree_off;
	int tl_corner;
	int fp0, fp1;

	//**************************************************************************
	//******Part#1)Load Feature Data into Shared Memory Collaboratively...******
	//**************************************************************************
	for(i = threadIdx.x; i<RFIC_INT_CONSTS[2]; i += blockDim.x)
		features[i] = forests_offs[i];
	__syncthreads();
	//**************************************************************************
	//***Part#2)Make Pixel Comparison and Store Result in "index" Variable...***
	//**************************************************************************
	for(i = start + blockDim.x * blockIdx.x + threadIdx.x; 
		i<end; 
		i += gridDim.x * blockDim.x) {
		bb_idx      = bb_sh_amount[i] + i;
		feature_ptr = bbox_offs[bb_idx + TLD_OBJ_INT_CONSTS[0] * PTR_TO_FEATURES];
		tl_corner   = abs_tl_corner[bb_idx];
		bb_idx      = i - start;
		for(j = 0; j<RFIC_INT_CONSTS[0]; j++) {
			tree_off = feature_ptr + j * 2 * RFIC_INT_CONSTS[1];
			index    = 0;
			for(k = 0; k<RFIC_INT_CONSTS[1]; k++) {
				//ILP...
				index <<= 1;
				fp0     = blurred_image[tl_corner + features[tree_off + 2 * k]];
				fp1     = blurred_image[tl_corner + features[tree_off + 2 * k + 1]];
				if(fp0 > fp1)
					index |= 1;
			}//End of Features...
			conf_indices[j * num_of_bb + bb_idx] = index;
		}//End of Forests...
	}//End of BBOXes...
}

__host__
void FastDetection::computePV(TOTAL_RECALL_COMP_STR *trc,
							  float min_var) {

	cudaError_t cuda_status;
	int i;
	int exec_off;
	//Patch Variance on GPU...
	for(i = 0; i<trc->num_of_async_inv; i++) {
		exec_off = i * TRC_EXEC_PARAMS_COUNT;
		computePVOnGPU<<<trc->exec_params[exec_off + TRC_EXEC_PV_TOTAL_NUM_OF_CL], trc->pv_comp_block_dim, trc->pv_comp_sm_size, trc->pv_streams[i]>>>
			(trc->dev_bbox_offs,
			 _dev_iis,
			 thrust::raw_pointer_cast(&(*trc->dev_cum_sum_of_scan_lines)[0]) + trc->exec_params[exec_off + TRC_EXEC_PV_INV_OFF],
			 thrust::raw_pointer_cast(&(*trc->dev_scan_line_rows)[0]),
			 thrust::raw_pointer_cast(&(*trc->dev_cum_sum_of_bb_per_cluster)[0]) + trc->exec_params[exec_off + TRC_EXEC_PV_INV_OFF],
		     min_var,
			 trc->dev_bb_var,
			 trc->dev_pv_status);
	}//End of for-Loop...
	//Stream Compaction on GPU...
	for(i = 0; i<trc->num_of_async_inv; i++) {
		exec_off = i * TRC_EXEC_PARAMS_COUNT;
		//Stream Compaction After PV-Computation...
		DeviceScan::InclusiveSum(trc->dev_sh_am_is_buf[i],
								 trc->sh_temp_buf_size[i],
							     trc->dev_pv_status + trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF],
								 trc->dev_bb_lsh + trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF],
							     trc->exec_params[exec_off + TRC_EXEC_PV_TOTAL_NUM_OF_BB],
								 trc->pv_streams[i]);
	}//End of for-Loop...
	for(i = 0; i<trc->num_of_async_inv; i++) {
		exec_off = i * TRC_EXEC_PARAMS_COUNT;
		findRightShifts<<<trc->st_comp_grid_dim[i], trc->st_comp_block_dim[i], (size_t)0, trc->pv_streams[i]>>>
			(trc->exec_params[exec_off + TRC_EXEC_PV_TOTAL_NUM_OF_BB],
			 trc->dev_pv_status + trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF],
			 trc->dev_bb_lsh + trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF],
			 trc->dev_bb_rsh + trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF]);
	}//End of for-Loop...
	//Copying Number of BBOXes That Have Passed PV-Test to Host...
	for(i = 0; i<trc->num_of_async_inv; i++) {
		exec_off = i * TRC_EXEC_PARAMS_COUNT;
		//Number of Remaining BBOXes After PV-Test...
		cudaMemcpyAsync((void*)(&trc->exec_params[exec_off + TRC_EXEC_CRFI_NUM_OF_REM_BB]),
						(void*)(trc->dev_bb_lsh + 
						           (trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF] + 
								    trc->exec_params[exec_off + TRC_EXEC_PV_TOTAL_NUM_OF_BB] - 1)),
						sizeof(int),
						cudaMemcpyDeviceToHost,
						trc->pv_streams[i]);
		cuda_status = cudaEventRecord(trc->pv_events[i], trc->pv_streams[i]);
		//cudaStreamQuery(trc->tr_streams[i]);//Flush the WDDM Buffer Immediately(Only On Windows Vista+ Platforms)...
		if(cuda_status != cudaSuccess)
			throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR),
							        "BB Variances(Copying)",
									cudaGetErrorString(cuda_status)),
							 fast_detection,
							 ER_FASTDET_CUDA_MEM_ERROR);
	}//End of for-Loop...
}

__host__
void FastDetection::measureBBOXOffsets(TOTAL_RECALL_COMP_STR *trc, 
									   int &exec_off,
									   int &thread_id,
									   int &start,
									   int &end,
									   int tpri_pos,
									   float *weights) {
	
	int i, j;
	double conf;
	int start_off;
	int num_of_rem_bb;

	start_off     = trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF] * trc->num_of_trees;
	num_of_rem_bb = trc->exec_params[exec_off + TRC_EXEC_CRFI_NUM_OF_REM_BB];
	for(i = start; i<end; i++) {
		conf = 0.0;
		for(j = 0; j<trc->num_of_trees; j++)
			conf += weights[j * trc->num_indices + trc->gpu_rfis[start_off + j * num_of_rem_bb + i]];
		trc->tpri_out_conf[thread_id][tpri_pos] = conf;
		tpri_pos++;
	}
}

__host__
void FastDetection::initializeTotalRecallComp(TOTAL_RECALL_COMP_STR *trc) {

	cudaError cuda_status;
	int consts[3];
	//Move it Without Casting!
	cuda_status = cudaMemcpyToSymbol(TLD_OBJ_INT_CONSTS, 
		                             &trc->num_of_bboxes, 
								     sizeof(int));
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR), 
								"TLD_OBJ_INT_CONSTS (in initializePVCOnGPU)", 
								cudaGetErrorString(cuda_status)), 
						 fast_detection, 
						 ER_FASTDET_CUDA_MEM_ERROR);
	//Move Constant Data to GPU for Computing EC-Indices...
	consts[0]   = trc->num_of_trees;
	consts[1]   = trc->num_of_features;
	consts[2]   = consts[0] * consts[1] * trc->num_of_sl * 2;
	cuda_status = cudaMemcpyToSymbol(RFIC_INT_CONSTS, consts, sizeof(int) * 3);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR), 
								"RFIC_INT_CONSTS (in constructor)", 
								cudaGetErrorString(cuda_status)), 
						 fast_detection, 
						 ER_FASTDET_CUDA_MEM_ERROR);
}

__host__
double FastDetection::computeTotalRecall(TOTAL_RECALL_COMP_STR *trc,
										 float min_var,
										 float *weights,
										 int *out_pattern,
										 float *out_conf,
										 float *out_bb_var) {

	cudaError_t stream_status;
	int jobs_done;
	int num_of_bb;
	int start_off;
	//Thread Private Fields...
	int i, j, k;
	int exec_off;
	int bb_idx;
	int thread_id;
	int start, end;
	//For Timing...
	__int64 counter_start; 
	double pc_freq;
	double total_time;
	//*******************************************************************************
	//************************Pre-Parallel-Initialization!...************************
	//*******************************************************************************
	initializeTotalRecallComp(trc);
	cudaFuncSetCacheConfig(computePVOnGPU, cudaFuncCachePreferShared);
	omp_set_dynamic(0);
	omp_set_num_threads(_num_of_cpu_threads);
	cudaFuncSetCacheConfig(computePVOnGPU, cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(computeRFIndicesOnGPU, cudaFuncCachePreferShared);
	startCounter(&counter_start, &pc_freq);
	#pragma omp parallel private(thread_id, i, j, k, bb_idx, exec_off, start, end)
	{
		//*******************************************************************************
		//**************************PV Computation on GPU Cores**************************
		//*******************************************************************************
		#pragma omp single
		{
			computePV(trc, min_var);
		}
		//*******************************************************************************
		//*********************************Initialize!...********************************
		//*******************************************************************************
		thread_id                      = omp_get_thread_num();
		trc->ready_for_copy[thread_id] = 0;
		//*******************************************************************************
		//*********************Ensemble Classifier On CPU/GPU Cores**********************
		//*******************************************************************************
		for(i = 0; i<trc->num_of_async_inv; i++) {
			exec_off = i * TRC_EXEC_PARAMS_COUNT;
			//Reset All Confidence Values...
			#pragma omp single 
			{
				stream_status = cudaEventSynchronize(trc->pv_events[i]);
				if(stream_status != cudaSuccess)
					throw HETLDError(format(GETFDETERROR(ER_FASTDET_STREAM_FAILED),
											i,
											"Computing Patch Variance",
											cudaGetErrorString(stream_status)),
									 fast_detection,
									 ER_FASTDET_STREAM_FAILED);
				trc->exec_params[exec_off + TRC_EXEC_CRFI_NUM_OF_REM_BB] += 
					trc->exec_params[exec_off + TRC_EXEC_PV_TOTAL_NUM_OF_BB];
				//*******************************************************************************
				//************************Calculate RFIs on GPU Cores...*************************
				//*******************************************************************************
				findKLPForBBOXProcessing(trc->exec_params[exec_off + TRC_EXEC_CRFI_NUM_OF_REM_BB],
									     trc->rfi_comp_grid_dim,
										 trc->rfi_comp_block_dim);
				computeRFIndicesOnGPU<<<trc->rfi_comp_grid_dim, trc->rfi_comp_block_dim, trc->rfi_comp_sm_size,  trc->crfi_streams[i]>>>
					(trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF],
					 trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF] + 
					     trc->exec_params[exec_off + TRC_EXEC_CRFI_NUM_OF_REM_BB],
					 trc->exec_params[exec_off + TRC_EXEC_CRFI_NUM_OF_REM_BB],
					 _mem_module->getDevBlurredCurFrame(),
					 trc->dev_bbox_offs,
					 trc->dev_abs_tl_corner,
					 trc->dev_forests_offs,
					 trc->dev_bb_rsh,
					 trc->dev_rfis + trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF] * trc->num_of_trees);
				cudaMemcpyAsync((void*)(trc->gpu_rfis + 
								            trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF] * trc->num_of_trees),
							    (void*)(trc->dev_rfis + 
									        trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF] * trc->num_of_trees),
						        sizeof(int) * trc->exec_params[exec_off + TRC_EXEC_CRFI_NUM_OF_REM_BB] * trc->num_of_trees,
						        cudaMemcpyDeviceToHost,
								trc->crfi_streams[i]);
				cudaEventRecord(trc->crfi_events[i], trc->crfi_streams[i]);
				//*******************************************************************************
				//***********************Copying Shift Amounts to Host...************************
				//*******************************************************************************
				cudaMemcpyAsync((void*)(trc->bb_rsh_amount + trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF]),
								(void*)(trc->dev_bb_rsh + trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF]),
								sizeof(int) * trc->exec_params[exec_off + TRC_EXEC_CRFI_NUM_OF_REM_BB],
								cudaMemcpyDeviceToHost,
								trc->pv_streams[i]);
				cudaEventRecord(trc->pv_events[i], trc->pv_streams[i]);
				//Compute Avg Size of the Chunk That will be Processed by Each CPU Thread...
				trc->exec_params[exec_off + TRC_EXEC_CCV_BB_PER_THREAD] = 
					(int)ceilf(trc->exec_params[exec_off + TRC_EXEC_CRFI_NUM_OF_REM_BB] / (float)_num_of_cpu_threads);
			}//End of single-Region for PVC and RFIC...
			//Partially Clear Confidence Values...
			#pragma omp single nowait
			{
				start = trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF];
				end   = start + trc->exec_params[exec_off + TRC_EXEC_PV_TOTAL_NUM_OF_BB];
				for(j = start; j<end; j++)
					out_conf[j] = 0.0;
			}//End of single-Region...
			//Calculate Thread-Private Values...
			start = thread_id * trc->exec_params[exec_off + TRC_EXEC_CCV_BB_PER_THREAD];
			end   = (thread_id == this->_num_of_cpu_threads - 1) ? 
					trc->exec_params[exec_off + TRC_EXEC_CRFI_NUM_OF_REM_BB] : 
					    start + trc->exec_params[exec_off + TRC_EXEC_CCV_BB_PER_THREAD];
			if(i < trc->num_of_async_inv - 1)
				trc->tpri_exec_params[thread_id][i + 1] = trc->tpri_exec_params[thread_id][i] + (end - start);
			//Synchronize on CUDA-Events...
			#pragma omp single 
			{
				//Synchronize on Event of Copying BB Shift Amounts...
				stream_status = cudaEventSynchronize(trc->pv_events[i]);
				if(stream_status != cudaSuccess)
					throw HETLDError(format(GETFDETERROR(ER_FASTDET_STREAM_FAILED),
											i,
											"Copying BB Shift Amounts from Device to Host",
											cudaGetErrorString(stream_status)),
									 fast_detection,
									 ER_FASTDET_STREAM_FAILED);
				//Synchronize on Event of Computing RFIs...
				stream_status = cudaEventSynchronize(trc->crfi_events[i]);
				if(stream_status != cudaSuccess)
					throw HETLDError(format(GETFDETERROR(ER_FASTDET_STREAM_FAILED),
											i,
											"CRFI Failed!",
											cudaGetErrorString(stream_status)),
									 fast_detection,
									 ER_FASTDET_STREAM_FAILED);
			}//End of single-Region for CUDA-Stream Synchronization...
			//It is now Time to Sum up Weights for Each BBOX...
			measureBBOXOffsets(trc,
							   exec_off,
							   thread_id,
							   start,
							   end,
							   trc->tpri_exec_params[thread_id][i],
							   weights);
		}//End of for-Loop...
		trc->ready_for_copy[thread_id] = 1;
		//*******************************************************************************
		//*****************Copying Patterns to Their Final Destination*******************
		//*******************************************************************************
		#pragma omp single nowait 
		{
			for(i = 0; i<trc->num_of_async_inv; i++) {
				exec_off  = i * TRC_EXEC_PARAMS_COUNT;
				num_of_bb = trc->exec_params[exec_off + TRC_EXEC_CRFI_NUM_OF_REM_BB];
				start     = trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF];
				end       = start + num_of_bb;
				start_off = start * trc->num_of_trees;
				for(j = start; j<end; j++) {
					bb_idx = trc->bb_idx[trc->bb_rsh_amount[j] + j] * trc->num_of_trees;
					for(k = 0; k<trc->num_of_trees; k++)
						out_pattern[bb_idx + k] = trc->gpu_rfis[start_off + k * num_of_bb + (j - start)];
				}//End of Inner-for-Loop...
			}//End of Outermost-for-Loop...
		}//End of single-Region for
		//*******************************************************************************
		//***********************Copying BB Variances if Asked **************************
		//*******************************************************************************
		#pragma omp single nowait 
		{
			if(out_bb_var != NULL) {
				cudaMemcpy((void*)trc->bb_var,
						   (void*)trc->dev_bb_var,
						   sizeof(float) * trc->num_of_bboxes,
						   cudaMemcpyDeviceToHost);
				for(i = 0; i<trc->num_of_bboxes; i++)
					out_bb_var[trc->bb_idx[i]] = trc->bb_var[i];
			}//End of if-Block...
		}//End of single-Region for
		//*******************************************************************************
		//*****************Copying Results to Their Final Destination********************
		//*********************Avoiding "False Cache-Line Sharing"***********************
		//*******************************************************************************
		#pragma omp single nowait
		{
			jobs_done = 0;
			while(1) {//Busy-Wait...
				if(jobs_done == this->_num_of_cpu_threads) //All Copying Process was Done and Thread is Exiting...
					break;
				thread_id = -1;
				for(i = 0; i<this->_num_of_cpu_threads; i++) {
					if(trc->ready_for_copy[i] == 1) {
						thread_id = i;
						break;
					}
				}//End of for-Loop...
				if(thread_id != -1) {
					//Copy Process...
					k = 0;
					for(i = 0; i<trc->num_of_async_inv; i++) {
						exec_off = i * TRC_EXEC_PARAMS_COUNT;
						start    = thread_id * trc->exec_params[exec_off + TRC_EXEC_CCV_BB_PER_THREAD];
					    end      = (thread_id == this->_num_of_cpu_threads - 1) ? 
								   trc->exec_params[exec_off + TRC_EXEC_CRFI_NUM_OF_REM_BB] : 
								       start + trc->exec_params[exec_off + TRC_EXEC_CCV_BB_PER_THREAD];
						for(j = start; j<end; j++) {
							bb_idx = trc->exec_params[exec_off + TRC_EXEC_PV_START_OFF] + j;
							out_conf[trc->bb_idx[bb_idx + trc->bb_rsh_amount[bb_idx]]] = 
								trc->tpri_out_conf[thread_id][k];
							k++;
						}//End of Inner-for-Loop...
					}//End of Outer-for-Loop...
					jobs_done++;
					trc->ready_for_copy[thread_id] = -1;
				}//End of if-Block...
			}//End of while-Loop...
		}//End of single Region(Copying Results to Their Final Destination)...
	}//End of parallel-Block...
	total_time = getCounter(&counter_start, &pc_freq);
	return total_time;
}

//********************************************************************************
//*******************Test Methods for Diagnostic Purposes...**********************
//********************************************************************************
#if PRODUCTION_ENV == 0
__host__
int* FastDetection::computeAllPV(TOTAL_RECALL_COMP_STR *trc, int min_var) {

	cudaError cuda_status;
	int i;
	int inv_offset;
	int total_num_of_cl;
	int *out_pv_status  = NULL;
	int *temp_pv_status = NULL;
	
	initializeTotalRecallComp(trc);
	cudaFuncSetCacheConfig(computePVOnGPU, cudaFuncCachePreferShared);
	//Invoke Test Kernel...
	for(i = 0; i<trc->num_of_async_inv; i++) {
		inv_offset      = i * trc->max_num_of_bpg;
		total_num_of_cl = trc->num_of_clusters - i * trc->max_num_of_bpg;
		total_num_of_cl = total_num_of_cl >  trc->max_num_of_bpg ? trc->max_num_of_bpg : total_num_of_cl;
		computePVOnGPU<<<total_num_of_cl, trc->pv_comp_block_dim, trc->pv_comp_sm_size>>>
			(trc->dev_bbox_offs, 
			 _dev_iis, 
			 thrust::raw_pointer_cast(&(*trc->dev_cum_sum_of_scan_lines)[0]) + inv_offset, 
			 thrust::raw_pointer_cast(&(*trc->dev_scan_line_rows)[0]), 
			 thrust::raw_pointer_cast(&(*trc->dev_cum_sum_of_bb_per_cluster)[0]) + inv_offset,
			 min_var,
			 trc->dev_bb_var,
			 trc->dev_pv_status);
		cuda_status = cudaGetLastError();
		if(cuda_status != cudaSuccess)
			throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_KER_FAILED), 
								    "computePVOnGPU", 
									cudaGetErrorString(cuda_status)), 
							 fast_detection, 
							 ER_FASTDET_CUDA_KER_FAILED);
	}//End of for-Loop...
	cudaDeviceSynchronize();
	temp_pv_status = (int*)malloc(sizeof(int) * trc->num_of_bboxes);
	out_pv_status  = (int*)malloc(sizeof(int) * trc->num_of_bboxes);
	cuda_status    = cudaMemcpy((void*)temp_pv_status,
							    (void*)trc->dev_pv_status,
						        sizeof(int) * trc->num_of_bboxes,
							    cudaMemcpyDeviceToHost);
	if(cuda_status != cudaSuccess)
		throw HETLDError(format(GETFDETERROR(ER_FASTDET_CUDA_MEM_ERROR), 
								"PV Status from Device to Host",
								cudaGetErrorString(cuda_status)), 
						 fast_detection, 
						 ER_FASTDET_CUDA_MEM_ERROR);
	for(i = 0; i<trc->num_of_bboxes; i++)
		out_pv_status[trc->bb_idx[i]] = temp_pv_status[i];
	//Free Allocated Resources...
	free((void*)temp_pv_status);
	return out_pv_status;
}

#endif

__host__
FastDetection::~FastDetection() {

	cudaFree((void*)_dev_iis);
}
