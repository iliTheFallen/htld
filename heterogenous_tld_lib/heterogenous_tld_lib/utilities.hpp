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
* utilities.hpp
*
* Author: Ilker GURCAN
*/

#ifndef UTILTIES_H_
#define UTILTIES_H_

#include<Windows.h>
#include<string>
#include<cstdarg>
#include "cuda.h"
#include "cuda_runtime.h"
#include "global_declarations.hpp"

#define round(fp) (int)((fp) >= 0 ? (fp) + 0.5 : (fp) - 0.5)

//********************************************************************************
//**************************Identifying Processor...******************************
//********************************************************************************
struct cpu_props {

	int numa_node_count;
	int physical_proc_pack_count;
	int physical_proc_cores;
	int logical_proc_cores;
	int num_of_l1_cache;
	int num_of_l2_cache;
	int num_of_l3_cache;
};

typedef struct cpu_props CPU_PROPS;

HETLD_API void getCPUProperties(CPU_PROPS* cpu_props);

//********************************************************************************
//****************************Measuring CPU Time...*******************************
//********************************************************************************
HETLD_API void startCounter(__int64 *counter_start, double *pc_freq);

HETLD_API double getCounter(__int64 *counter_start, double *pc_freq);

//********************************************************************************
//**************************CUDA Device Properties...*****************************
//********************************************************************************
HETLD_API int getNumOfDevices();

HETLD_API cudaDeviceProp* getDeviceProps(int dev_num);

HETLD_API int getNumOfCUDACoresPerSM(int dev_num);
//Num of Threads Per Block With Respect to SM's Max # of Threads...
HETLD_API int findOptimalNumOfTPB(int dev_num);

//********************************************************************************
//******************************I/O Operations...*********************************
//********************************************************************************
HETLD_API void dispatchOSToFile(const std::string &file_name);

HETLD_API void resetOutputStream();

HETLD_API std::string format(const char* format, ...);

#endif /*UTILTIES_H_*/
