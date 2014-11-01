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
* global_declarations.hpp
*
* Author: Ilker GURCAN
*/

#ifdef HETLD_EXPORTS
#define HETLD_API __declspec(dllexport) 
#else
#define HETLD_API __declspec(dllimport) 
#endif

#ifndef GLOBALDECLARATIONS_H_
#define GLOBALDECLARATIONS_H_

#define PRODUCTION_ENV 0

//********************************************************************************
//**********************C++ Modules For heterogenous_tld_lib...*******************
//********************************************************************************
enum HETLDModules {
master         = 0,
mem_management = 1,
fast_tracking  = 2,
fast_detection = 3,
tld_object     = 4
};

//It Should Be Static Since It Is Being Included By Several Source Files...
static const char* module_tags[] = {"master", "mem_management", "fast_tracking", "fast_detection"};

#define GETMODULETAG(module) module_tags[module]

//********************************************************************************
//*************************Required Custom Data Types...**************************
//********************************************************************************
enum Bool {
	T = 1,
	F = 0
};

typedef unsigned char ubyte;

#endif /* GLOBALDECLARATIONS_H_ */
