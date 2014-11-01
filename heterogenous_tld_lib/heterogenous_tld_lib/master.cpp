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
* master.cpp
* 
* Author: Ilker GURCAN
*/


//TODO Check for Whether Enumerated GPUs Have Enough Capabilities to Execute...
//TODO Assign Each New Object to Track to a Different GPU in Round-Robin Fashion...
#define HETLD_EXPORTS

#include "master.hpp"
#include "hetld_errors.hpp"


Master::Master(int f_width, 
	           int f_height,
			   bool is_tracking_enabled,
			   bool is_detection_enabled,
			   double sigma_of_gaussian_blur) {


	if(!is_tracking_enabled && !is_detection_enabled)
		throw HETLDError(format(GETMASTERERROR(ER_MASTER_AT_LEAST_ONE_ACTIVE_MODULE)), 
						 master,
						 ER_MASTER_AT_LEAST_ONE_ACTIVE_MODULE);
	//Find CPU Properties...
	_cpu_props = (CPU_PROPS*)malloc(sizeof(CPU_PROPS));
	getCPUProperties(_cpu_props);
	//Create Modules...
	_mem_module = new MemoryManagement(f_width, f_height, sigma_of_gaussian_blur);

	if(is_tracking_enabled)
		_fast_tr = new FastTracking(_mem_module);
	else 
		_fast_tr = nullptr;

	if(is_detection_enabled)
		_fast_det = new FastDetection(_mem_module, _cpu_props);
	else 
		_fast_det = nullptr;
	//TLD-Object Map...
	_num_of_objects = 0;
	_tld_object_map = new TLD_OBJECT_MAP();
}

int Master::createTLDObject(FAST_TR_INIT_PARAMS *fti, 
	                        FAST_DET_INIT_PARAMS *fdi) {

	TLDObject *obj = new TLDObject(_fast_tr != nullptr ? true : false, 
								   _fast_det != nullptr ? true : false);
	
	//********************************************************************************
	//****************Initializing Data Structure for Fast Tracking...****************
	//********************************************************************************
	if(_fast_tr != nullptr && fti != NULL) {
		_fast_tr->initializeTLDObject(obj->getFastTrStr(), 
									  fti->use_initial_flows,
									  fti->win_size,
									  fti->iters,
									  fti->max_pyr_level);
		//Print Out Debug Info for This TLD Object...
		obj->printFTS();
	}

	//********************************************************************************
	//***************Initializing Data Structure for Fast Detection...****************
	//********************************************************************************
	if(_fast_det != nullptr && fdi != NULL) {
		_fast_det->initializeTLDObject(obj->getFastDetStr(),
									   fdi->num_of_sl,
									   fdi->num_of_trees,
									   fdi->num_of_features,
									   fdi->num_of_bboxes_per_sl,
									   fdi->num_of_lrb_per_sl,
									   fdi->bbox_offs,
									   fdi->forest_offs);
		//Print Out Debug Info for This TLD Object...
		obj->printFDS();
	}

	//Add to Object Map...
	_tld_object_map->insert(TLD_OBJECT_PAIR(obj->getObjectId(), obj));
	_num_of_objects++;

	return obj->getObjectId();
}

void Master::destroyTLDObject(int object_id) {

	TLDObject *obj = nullptr;
	TLD_OBJECT_MAP::iterator res = _tld_object_map->find(object_id);
	if(res == _tld_object_map->end()) {
		fprintf(stderr, 
			    GETMASTERERROR(ER_MASTER_NO_SUCH_TLD_OBJ),
			    object_id);
		return;
	}

	obj = res->second;
	//Free All Allocated Resources for Modules...
	if(_fast_tr != nullptr)
		_fast_tr->finalizeTLDObject(obj->getFastTrStr());
	if(_fast_det != nullptr)
		_fast_det->finalizeTLDObject(obj->getFastDetStr());
	//Finally Free Object Itself...
	_tld_object_map->erase(object_id);
	delete obj;
	_num_of_objects--;
}

TLDObject* Master::getTLDObject(int object_id) {

	TLD_OBJECT_MAP::iterator res = _tld_object_map->find(object_id);

	if(res == _tld_object_map->end())
		return nullptr;
	else 
		return res->second;
}

Master::~Master(void) {

	while(_tld_object_map->size() > 0)
		destroyTLDObject(_tld_object_map->begin()->first);
	delete _tld_object_map;
	free((void*)_cpu_props);
	if(_fast_tr != nullptr)
		delete _fast_tr;
	if(_fast_det != nullptr)
		delete _fast_det;
	delete _mem_module;
}
