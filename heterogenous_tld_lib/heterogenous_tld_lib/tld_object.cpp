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
* tld_object.cpp
* 
* Author: Ilker GURCAN
*/

#define HETLD_EXPORTS

#include<iostream>
#include<stdlib.h>
#include "hetld_errors.hpp"
#include "utilities.hpp"
#include "tld_object.hpp"

int TLDObject::_obj_seq = -1;

TLDObject::TLDObject(bool is_tracking_enabled,
			         bool is_detection_enabled) {

	_object_id = ++TLDObject::_obj_seq;
	if(is_tracking_enabled)
		_fast_tr_str = (FAST_TRACKING_STR*)malloc(sizeof(FAST_TRACKING_STR));
	else 
		_fast_tr_str = NULL;
	if(is_detection_enabled)
		_fast_det_str = (FAST_DETECTION_STR*)malloc(sizeof(FAST_DETECTION_STR));
	else 
		_fast_det_str = NULL;
}

void TLDObject::printFTS() {

	if(_fast_tr_str == NULL)
		throw HETLDError(format(GETTLDOBJERROR(ER_TLDOBJ_MODULE_NOT_ENABLED), 
								"fast_tracking", 
								"printFTS", 
								_object_id),
						 tld_object,
						 ER_TLDOBJ_MODULE_NOT_ENABLED);
	std::cout<<"*******Parameters for Tracking of the TLD Object with Id : "<<_object_id<<std::endl;
	std::cout<<"Win Size : "<<_fast_tr_str->lk->winSize.width<<" / "<<_fast_tr_str->lk->winSize.height<<std::endl;
	std::cout<<"Num of Iterations : "<<_fast_tr_str->lk->iters<<std::endl;
	std::cout<<"Num of Levels : "<<_fast_tr_str->lk->maxLevel<<std::endl;
	std::cout<<"Is Use Initial Flows : ";
	if(_fast_tr_str->lk->useInitialFlow)
		std::cout<<"true"<<std::endl;
	else
		std::cout<<"false"<<std::endl;
}

void TLDObject::printFDS() {

	if(_fast_det_str == NULL)
		throw HETLDError(format(GETTLDOBJERROR(ER_TLDOBJ_MODULE_NOT_ENABLED), 
								"fast_detection", 
								"printFDS", 
								_object_id),
						 tld_object,
						 ER_TLDOBJ_MODULE_NOT_ENABLED);

	TOTAL_RECALL_COMP_STR *trc = _fast_det_str->trc;
	std::cout<<"****Parameters for Detection of the TLD Object with Id : "<<_object_id<<std::endl;
	std::cout<<"Num Of BBOXes : "<<trc->num_of_bboxes<<std::endl;
	std::cout<<"Num Of Clusters : "<<trc->num_of_clusters<<std::endl;
	std::cout<<"Maximum Num Of Blocks Per Grid : "<<trc->max_num_of_bpg<<std::endl;
	std::cout<<"Num Of Threads Per Block : "
		     <<(trc->pv_comp_block_dim.x * trc->pv_comp_block_dim.y)
			 <<std::endl;
	std::cout<<"Shared Memory Size : "<<trc->pv_comp_sm_size<<std::endl;
	std::cout<<"Num Of Asynchronous Invocations : "<<trc->num_of_async_inv<<std::endl;
	std::cout<<"***************************************\n"<<std::endl;
}

TLDObject::~TLDObject(void) {

	if(_fast_tr_str != NULL)
		free((void*)_fast_tr_str);
	if(_fast_det_str != NULL)
		free((void*)_fast_det_str);
}
