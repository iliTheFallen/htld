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
* hetld_macros.hpp
* 
* Author: Ilker GURCAN
*/

#ifndef HETLDMACROS_HPP_
#define HETLDMACROS_HPP_

#ifdef ARCH_3X
#define MAX_NUM_OF_BLOCKS_PER_SM 8
#else
#define MAX_NUM_OF_BLOCKS_PER_SM 4
#endif

#define WARP_SIZE 32
#define MIN_OCCUPANCY_PERC 0.50 //50%
#define MIN_NUM_OF_TPB 64 //Don't Change!...
#define NUM_OF_CYCLES_FOR_RL 24 //Don't Change!...

//Attributes in Order : 
//Top-Left Corner
//Bottom-Left Corner
//Top-Right Corner
//Bottom-Right Corner
//Area of Scanning Window
//Pointer to Features for That Scale
//Number of Left-Right Boxes
#define NUM_OF_BBOX_ATTRS_ON_CPU 6
//Attributes in Order : 
//Top-Left Corner
//Bottom-Left Corner
//Top-Right Corner
//Bottom-Right Corner
//Area of Scanning Window
#define NUM_OF_BBOX_ATTRS_ON_GPU 6
#define TOP_LEFT_CORNER 0
#define BOTTOM_LEFT_CORNER 1
#define TOP_RIGHT_CORNER 2
#define BOTTOM_RIGHT_CORNER 3
#define PTR_TO_FEATURES 4
#define AREA_OF_SCANNING_WINDOW 5

#endif /* HETLDMACROS_HPP_ */
