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
* fast_detection.cuh
* 
* Author: Ilker GURCAN
*/

#pragma once
//TODO Don't Forget to Create Reference Variables(Aliases) for All Method Parameters
#include<stdlib.h>
#include<vector>
#include<set>
#include<map>
#include<cuda_runtime_api.h>
#include<nppdefs.h>
#include<thrust\device_vector.h>
#include<thrust\host_vector.h>
#include "utilities.hpp"
#include "imodule.hpp"
#include "mem_management.cuh"

using namespace thrust;

/**
* "EC" Stands for "Ensemble Classifier".
* "CL" Stands for "Cluster"
* "RFI" Stands for "Calculating Random Forest Indices".
* "PV" Stands for "Patch Variance".
* "REM" Stands for "Remaining".
* "CCV" Stands for "Calculating Confidence Values".
* "RCV" Stands for "Reset Confidence Values".
*/

//Ensemble Classifier Execution Parameters Per Async Kernel Inv...
#define TRC_EXEC_PARAMS_COUNT 6
//for Computing PVs on GPU...
#define TRC_EXEC_PV_INV_OFF 0//inv_offset
#define TRC_EXEC_PV_START_OFF 1//start_offset for BBOXes
#define TRC_EXEC_PV_TOTAL_NUM_OF_BB 2//total_num_of_bb
#define TRC_EXEC_PV_TOTAL_NUM_OF_CL 3//total # of Clusters
//for Calculating Confidence Values and RFIs on CPU...
#define TRC_EXEC_CRFI_NUM_OF_REM_BB 4 //# of Remaining BBOXes after Running PV-Test
#define TRC_EXEC_CCV_BB_PER_THREAD 5 //Avg # of Confidence Values to be Calculated by Each Thread

#define TPRI_TRC_EXEC_PARAMS_COUNT 1
#define TPRI_TRC_CCV_START_OFF 0

struct total_recall_comp_str {

	//**********************************************************************
	//*********************Total Recall Computation...**********************
	//**********************************************************************
	int num_of_sl;
	int num_indices;
	int num_of_trees;
	int num_of_features;
	int num_of_bboxes;
	int num_of_clusters;
	device_vector<int> *dev_cum_sum_of_scan_lines;//Cumulative Sum of # of Scan Lines to Read into Shared Memory per Cluster in Order...
	device_vector<int> *dev_scan_line_rows;//Row Indices of Scan Lines for Each Cluster in Order...
	device_vector<int> *dev_cum_sum_of_bb_per_cluster;//Cumulative Sum of # of BBOXes per Cluster in Order...
	host_vector<int> *cum_sum_of_bb_per_cluster;
	int *dev_forests_offs;
	int *dev_bbox_offs;//SoA Access Patterns...
	int *dev_abs_tl_corner;//Since dev_bbox_offs's Offset Values are Relative to SM Addresses Within Blocks; That Field is Necessity.
	//Memory Fields That Should be Allocated Once for Performance Gain...
	//On Device Side...
	float *dev_bb_var;
	int *dev_pv_status;
	void **dev_sh_am_is_buf;
	int *dev_bb_lsh;
	int *dev_bb_rsh;
	int *dev_rfis;
	//On Host Side...
	float *bb_var;
	cudaStream_t *pv_streams;
	cudaEvent_t *pv_events;
	cudaStream_t *crfi_streams;
	cudaEvent_t *crfi_events;
	int *bb_idx;
	int *bb_rsh_amount;
	int *gpu_rfis;
	int *exec_params;
	//Thread Private Data...
	float **tpri_out_conf;
	int **tpri_exec_params;
	int *ready_for_copy;
	//Kernel Launch Params...
	//pv_computation...
	size_t pv_comp_sm_size;
	int max_num_of_bpg;
	int num_of_async_inv;
	dim3 pv_comp_block_dim;
	//st_compaction...
	size_t *sh_temp_buf_size;
	int *st_comp_grid_dim;
	int *st_comp_block_dim;
	//rfi_calculation...
	int rfi_comp_grid_dim; 
	int rfi_comp_block_dim;
	size_t rfi_comp_sm_size;
};

typedef struct total_recall_comp_str TOTAL_RECALL_COMP_STR;

//Data Structure for Fast Detection on CPU/GPU Cores Heterogeneously...
struct fast_detection_str {

	TOTAL_RECALL_COMP_STR *trc;
};

typedef struct fast_detection_str FAST_DETECTION_STR;

//********************************************************************************
//***********************Internally Used Data Structures...***********************
//********************************************************************************
typedef std::set<int> SCAN_LINE_SET;
//Set of Scan Lines With Shared Memory Indices
typedef std::map<int, int> SL_MAP;
typedef std::pair<int, int> SL_PAIR;

class HETLD_API FastDetection : public IModule
{
public:
	__host__
	FastDetection(MemoryManagement *mem_module,
				  CPU_PROPS* cpu_props);
	__host__
	enum HETLDModules getModule() {return fast_detection;};
	__host__
	virtual ~FastDetection();
	__host__
	void initializeTLDObject(FAST_DETECTION_STR *fast_det,
							 int num_of_sl,
							 int num_of_trees,
							 int num_of_features,
		                     int *num_of_bboxes_per_sl,
							 int *num_of_lrb_per_sl,
							 int *bbox_offs,
							 int *forest_offs);
	__host__
	void finalizeTLDObject(FAST_DETECTION_STR *fast_det);
	//API Methods...
	__host__
	void computeIIs(bool is_move_to_host);
	__host__
	double computeTotalRecall(TOTAL_RECALL_COMP_STR *trc,
						      float min_var,
							  float *weights,
							  int *out_pattern,
							  float *out_conf,
							  float *out_bb_var);
#if PRODUCTION_ENV == 0
	__host__
	int* computeAllPV(TOTAL_RECALL_COMP_STR *trc, int min_var);
#endif
protected:
	__host__
	void initializeTotalRecallComp(TOTAL_RECALL_COMP_STR *trc);
	__host__
	void findKLPForBBOXProcessing(int num_of_bboxes, 
	                              int &grid_dim,
	                              int &block_dim);
private:
	MemoryManagement  *_mem_module;
	Npp32s           **_dev_iis;
	int                _num_of_cpu_threads;
	//Private Methods...
	__host__
	void computePV(TOTAL_RECALL_COMP_STR *trc,
				   float min_var);
	__host__
	void measureBBOXOffsets(TOTAL_RECALL_COMP_STR *trc,
	                        int &exec_off,
						    int &thread_id,
						    int &start,
						    int &end,
							int tpri_pos,
						    float *weights);
};
