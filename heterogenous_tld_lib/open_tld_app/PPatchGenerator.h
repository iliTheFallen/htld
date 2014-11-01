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
* with the one from Surrey University for TLD Algorithm developed by Zdenek Kalal,
* and from Austrian Institute of Technology for C++ port coded by Georg Nebehay.
* If not, see <http://www.gnu.org/licenses/>. 
* Please contact Alptekin TEMIZEL for more info about 
* licensing atemizel@metu.edu.tr.
*
*/

/*
* PPatchGenerator.h
* 
* Author: Ilker GURCAN
*/

#ifndef PPATCHGENERATOR_H_
#define PPATCHGENERATOR_H_

#include<vector>
#include<opencv2/core/core.hpp>


void bbHull(int *windowOffsets, 
			std::vector<int> &indices, 
			int imW, 
			int *hull);

void extractPatch(cv::Mat &img, 
				  int *hull,
				  int bbW,
				  int bbH,
				  unsigned char filledColor,
				  cv::RNG &rng,
				  double noise, 
				  double angle, 
				  double shift, 
				  double scale, 
				  cv::Mat & noiseM,
				  cv::Mat &result);

#endif