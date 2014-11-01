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

#define HETLD_EXPORTS

#include<iostream>
#include<fstream>
#include<map>
#include "utilities.hpp"
#include "helper_cuda.h"

typedef std::map<int, cudaDeviceProp*> dev_props_map;
typedef std::pair<int, cudaDeviceProp*> dev_props_pair;

static dev_props_map* _dev_props = NULL;

//********************************************************************************
//****************************Measuring CPU Time...*******************************
//********************************************************************************
void startCounter(__int64 *counter_start, double *pc_freq) {
	
	LARGE_INTEGER li;
	if(!QueryPerformanceFrequency(&li)) 
		std::cout<<"QueryPerformanceFrequency Failed!"<<std::endl;
	
	*pc_freq = ((double)li.QuadPart)/1000.0;

	QueryPerformanceCounter(&li);
	*counter_start = li.QuadPart;
}

double getCounter(__int64 *counter_start, double *pc_freq) {
	
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return ((double)(li.QuadPart - (*counter_start))) / (*pc_freq);
}

//********************************************************************************
//**************************Identifying Processor...******************************
//********************************************************************************
typedef BOOL (WINAPI *LPFN_GLPI)(
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, 
    PDWORD);


// Helper function to count set bits in the processor mask.
DWORD CountSetBits(ULONG_PTR bitMask)
{
    DWORD LSHIFT = sizeof(ULONG_PTR)*8 - 1;
    DWORD bitSetCount = 0;
    ULONG_PTR bitTest = (ULONG_PTR)1 << LSHIFT;    
    DWORD i;
    
    for (i = 0; i <= LSHIFT; ++i)
    {
        bitSetCount += ((bitMask & bitTest)?1:0);
        bitTest/=2;
    }

    return bitSetCount;
}

void getCPUProperties(CPU_PROPS* cpu_props) {

	LPFN_GLPI glpi;
    BOOL done = FALSE;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = NULL;
    DWORD returnLength = 0;
    DWORD logicalProcessorCount = 0;
    DWORD numaNodeCount = 0;
    DWORD processorCoreCount = 0;
    DWORD processorL1CacheCount = 0;
    DWORD processorL2CacheCount = 0;
    DWORD processorL3CacheCount = 0;
    DWORD processorPackageCount = 0;
    DWORD byteOffset = 0;
    PCACHE_DESCRIPTOR Cache;

    glpi = (LPFN_GLPI) GetProcAddress(
                            GetModuleHandle(TEXT("kernel32")),
                            "GetLogicalProcessorInformation");
    if (NULL == glpi) 
    {
		std::cerr<<"\nGetLogicalProcessorInformation is not supported.\n"<<std::endl;
        return;
    }

    while (!done)
    {
        DWORD rc = glpi(buffer, &returnLength);

        if (FALSE == rc) 
        {
            if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) 
            {
                if (buffer) 
                    free(buffer);

                buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(
                        returnLength);

                if (NULL == buffer) 
                {
					std::cerr<<"\nError: Allocation failure\n"<<std::endl;
                    return;
                }
            } 
            else 
            {
				std::cerr<<"\nError : "<<GetLastError()<<std::endl;
                return;
            }
        } 
        else
        {
            done = TRUE;
        }
    }

    ptr = buffer;
    while (byteOffset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= returnLength) 
    {
        switch (ptr->Relationship) 
        {
        case RelationNumaNode:
            // Non-NUMA systems report a single record of this type.
            numaNodeCount++;
            break;

        case RelationProcessorCore:
            processorCoreCount++;

            // A hyperthreaded core supplies more than one logical processor.
            logicalProcessorCount += CountSetBits(ptr->ProcessorMask);
            break;

        case RelationCache:
            // Cache data is in ptr->Cache, one CACHE_DESCRIPTOR structure for each cache. 
            Cache = &ptr->Cache;
            if (Cache->Level == 1)
            {
                processorL1CacheCount++;
            }
            else if (Cache->Level == 2)
            {
                processorL2CacheCount++;
            }
            else if (Cache->Level == 3)
            {
                processorL3CacheCount++;
            }
            break;

        case RelationProcessorPackage:
            // Logical processors share a physical package.
            processorPackageCount++;
            break;

        default:
			std::cerr<<"\nError: Unsupported LOGICAL_PROCESSOR_RELATIONSHIP value.\n"<<std::endl;
            break;
        }
        byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        ptr++;
    }
    
    free(buffer);

	//Set CPU Properties...
	cpu_props->numa_node_count          = (int)numaNodeCount;
	cpu_props->physical_proc_pack_count = (int)processorPackageCount;
	cpu_props->physical_proc_cores      = (int)processorCoreCount;
	cpu_props->logical_proc_cores       = (int)logicalProcessorCount;
	cpu_props->num_of_l1_cache          = (int)processorL1CacheCount;
	cpu_props->num_of_l2_cache          = (int)processorL2CacheCount;
	cpu_props->num_of_l3_cache          = (int)processorL3CacheCount;
}

//********************************************************************************
//**************************CUDA Device Properties...*****************************
//********************************************************************************
int getNumOfDevices() {

	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	return device_count;
}

cudaDeviceProp* getDeviceProps(int dev_num) {

	cudaDeviceProp* dev_prop;
	cudaError_t status;

	if(_dev_props == NULL)
		_dev_props = new dev_props_map();
	//Device Has Not Been Queried Yet...
	if(_dev_props->find(dev_num) == _dev_props->end()) {
		dev_prop = (cudaDeviceProp*)malloc(sizeof(cudaDeviceProp));
		status   = cudaGetDeviceProperties(dev_prop, dev_num);
		if(status != cudaSuccess) {
			fprintf(stderr, 
				    "Cuda Device Query for Device %d Has Failed!",
					dev_num);
			return NULL;
		} else
			_dev_props->insert(dev_props_pair(dev_num, dev_prop));
	} else 
		dev_prop = _dev_props->find(dev_num)->second;
	return dev_prop;
}

int getNumOfCUDACoresPerSM(int dev_num) {

	int num_of_cores;
	cudaDeviceProp* dev_prop = _dev_props != NULL ? 
		                       _dev_props->find(dev_num)->second : 
	                           getDeviceProps(dev_num);

	if(dev_prop != NULL) {
		num_of_cores = _ConvertSMVer2Cores(dev_prop->major, dev_prop->minor);
	} else { 
		num_of_cores = -1;
		fprintf(stderr, 
				"getNumOfCUDACoresPerSM for Device %d Has Failed!\n No Such Device!",
				dev_num);
	}

	return num_of_cores;
}

int findOptimalNumOfTPB(int dev_num) {

	int n;
	cudaDeviceProp* props = getDeviceProps(dev_num);

	n = props->maxThreadsPerBlock;
	while(n >= 64) {
		if(props->maxThreadsPerMultiProcessor % n == 0) 
			break;
		else 
			n -= 32;
	}//End of while-Loop...

	return n;
}

//********************************************************************************
//******************************I/O Operations...*********************************
//********************************************************************************
static std::ofstream  *_out_file = nullptr;
static std::streambuf *_old_buf  = nullptr;

void dispatchOSToFile(const std::string &file_name) {

	_out_file = new std::ofstream(file_name, std::ios::out);
	_old_buf  = std::cout.rdbuf();
	std::cout.rdbuf(_out_file->rdbuf());
}

void resetOutputStream() {

	if(_out_file == nullptr)
		return;
	std::cout.rdbuf(_old_buf);
	_out_file->close();
	delete _out_file;
	_out_file = nullptr;
}

std::string format(const char* format, ...) {

	int size     = 512;
    char* buffer = 0;
    buffer       = new char[size];
    va_list vl;
    va_start(vl, format);
    int nsize = vsnprintf(buffer, size, format, vl);
    if(size<=nsize){ //fail delete buffer and try again
        delete[] buffer;
        buffer = 0;
        buffer = new char[nsize+1]; //+1 for /0
        nsize  = vsnprintf(buffer, size, format, vl);
    }
    std::string ret(buffer);
    va_end(vl);
    delete[] buffer;
    return ret;
}
