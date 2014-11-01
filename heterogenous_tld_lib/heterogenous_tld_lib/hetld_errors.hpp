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
* hetld_errors.hpp
* 
* Author: Ilker GURCAN
*/

#ifndef HETLDERRORS_H_
#define HETLDERRORS_H_

#include<stdexcept>
#include "global_declarations.hpp"

#define HETLD_NO_ERROR -1

//********************************************************************************
//*****************************Memory Management Module***************************
//********************************************************************************
#define ER_MEM_MAN_CUDA_MALLOC 0
#define ER_MEM_MAN_CUDA_FREE 1
#define ER_MEM_MAN_DEV_PTR_NULL 2
#define ER_MEM_MAN_HOST_PTR_NULL 3
#define ER_MEM_MAN_CUDA_MEM_CPY 4
#define ER_MEM_MAN_NPPI_MEM_CPY 5
#define ER_MEM_MAN_CUDA_KER_FAILED 6
#define ER_MEM_MAN_NOT_SUFFICIENT_SPACE 7
#define ER_MEM_MAN_KERNEL_EXEC_CONFIG_PR 8

static const char* mem_error_strs[] = {"Allocating Data for %s is Failed in Method %s\n Cuda Error : %s\n", 
                                       "While Deallocating Data for %s; An Error Occured in Method %s\n Cuda Error : %s\n", 
								       "Device Pointer That Was Supposed to Be Allocated for %s is Null!\n", 
                                       "Host Pointer That Was Supposed to Be Allocated for %s is Null!\n", 
                                       "Copying %s Data from %s to %s Failed!\n Cuda Error : %s\n", 
                                       "Copying %s Data from %s to %s Failed!\n NPPI Error Code : %d\n", 
                                       "Kernel Invocation for %s Has Failed!\n Cuda Error : %s\n",
                                       "There is not Sufficient Space on RAM to Allocate for Creating %s\n", 
                                       "A Convenient Kernel Launch Parameters for %s Cannot Be Found!\n" };

#define GETMEMERROR(code) mem_error_strs[code]

//********************************************************************************
//*******************************Fast Tracking Module*****************************
//********************************************************************************
#define ER_FASTTR_TLD_TR_OBJ_NULL 0

static const char *fasttr_error_strs[] = {"TLD Object's Tracking Structure is NULL!\nMethod : %s"};

#define GETFTRERROR(code) fasttr_error_strs[code]

//********************************************************************************
//******************************Fast Detection Module*****************************
//********************************************************************************
#define ER_FASTDET_SQR_INT_IMG_COMP 0
#define ER_FASTDET_NO_BBOX 1
#define ER_FASTDET_CUDA_MEM_ERROR 2
#define ER_FASTDET_CUDA_KER_FAILED 3
#define ER_FASTDET_GPU_RUN_OUT_OF_MEM 4
#define ER_FASTDET_NOT_SUFFICIENT_GPU_RES 5
#define ER_FASTDET_STREAM_FAILED 6

static const char *fastdet_error_strs[] = {"NPPI SqrIntegralImage Method Failed When It is Called in Method %s.\n NPPI Error Code : %d", 
                                           "No Bounding Box Has Been Passed to Initialization Step of fast_fern Module!",
                                           "CUDA Memory Error Occured While Processing Data for : %s\n Cuda Error : %s\n", 
                                           "Kernel Invocation for %s Has Failed!\n Cuda Error : %s\n", 
                                           "Ran Out of Memory While %s\n Reason : %s\n", 
                                           "Not Sufficient GPU Resources to Execute %s", 
                                           "Running Commands in Stream %d for Operation %s Has Failed!\n Cuda Error : %s\n"};

#define GETFDETERROR(code) fastdet_error_strs[code]

//********************************************************************************
//**********************************Master Module*********************************
//********************************************************************************
#define ER_MASTER_NO_SUCH_TLD_OBJ 0
#define ER_MASTER_AT_LEAST_ONE_ACTIVE_MODULE 1

static const char* master_error_strs[] = {"No TLD Object With Id %d Exists!\n", 
                                          "At Least One Module Has to Be Activated!"};

#define GETMASTERERROR(code) master_error_strs[code]

//********************************************************************************
//***********************************TLD Object **********************************
//********************************************************************************
#define ER_TLDOBJ_MODULE_NOT_ENABLED 0

static const char* tld_obj_error_strs[] = {"%s Module is not Enabled! You can't Perform %s on Object %d"};

#define GETTLDOBJERROR(code) tld_obj_error_strs[code]

//********************************************************************************
//*********************************HETLD Exception********************************
//********************************************************************************
class HETLDError : public std::runtime_error {

public:
	HETLDError(const std::string& what_arg, 
		       enum HETLDModules module, 
			   int error_code)
	:std::runtime_error(what_arg) {

		_module     = module;
		_error_code = error_code;
	};
	int _error_code;
	enum HETLDModules _module;
};

#endif /* HETLDERRORS_H_ */
