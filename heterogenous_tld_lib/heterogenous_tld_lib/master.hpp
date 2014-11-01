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
* master.hpp
* 
* Author: Ilker GURCAN
*/

#pragma once

#include<map>
#include "imodule.hpp"
#include "mem_management.cuh"
#include "fast_detection.cuh"
#include "fast_tracking.cuh"
#include "tld_object.hpp"
#include "utilities.hpp"

//********************************************************************************
//********************************TLDObject Map...********************************
//********************************************************************************

typedef std::map<int, TLDObject*> TLD_OBJECT_MAP;
typedef std::pair<int, TLDObject*> TLD_OBJECT_PAIR;

//********************************************************************************
//***************Initialization Params for fast_detection Module...***************
//********************************************************************************
struct fast_det_init_params {
	
	int num_of_sl;
	int num_of_trees;
	int num_of_features;
	int *num_of_bboxes_per_sl;
	int *num_of_lrb_per_sl;
	int *bbox_offs;
	int *forest_offs;
};

typedef struct fast_det_init_params FAST_DET_INIT_PARAMS;

//********************************************************************************
//****************Initialization Params for fast_tracking Module...***************
//********************************************************************************
struct fast_tr_init_params {

	bool use_initial_flows;
	cv::Size win_size;
	int iters;
	int max_pyr_level;
};

typedef struct fast_tr_init_params FAST_TR_INIT_PARAMS;

class HETLD_API Master : public IModule
{
public:
	Master(int f_width,
		   int f_height,
		   bool is_tracking_enabled,
		   bool is_detection_enabled,
		   double sigma_of_gaussian_blur = 2.0);
	virtual ~Master(void);
	enum HETLDModules getModule() {return master;};
	//Getters...
	FastDetection* getFastDet() {return _fast_det;};
	FastTracking* getFastTr() {return _fast_tr;};
	MemoryManagement* getMemModule() {return _mem_module;};
	//**********************************************************
	//****************TLDObject Related Methods*****************
	//**********************************************************
	int createTLDObject(FAST_TR_INIT_PARAMS *fti, 
		                FAST_DET_INIT_PARAMS *fdi);
	void destroyTLDObject(int object_id);
	int getNumOfObjects() {return _num_of_objects;};
	TLDObject* getTLDObject(int object_id);
private:
	CPU_PROPS        *_cpu_props;
	MemoryManagement *_mem_module;
	FastDetection    *_fast_det;
	FastTracking     *_fast_tr;
	int               _num_of_objects;
	TLD_OBJECT_MAP   *_tld_object_map;
};
