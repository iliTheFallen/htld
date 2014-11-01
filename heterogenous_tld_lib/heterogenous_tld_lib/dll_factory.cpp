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
* dll_factory.cpp
* 
* Author: Ilker GURCAN
*/

#define HETLD_EXPORTS

#include "dll_factory.hpp"

Master* createHETLDMasterModule(int f_width,
	                            int f_height,
								double sigma_of_gaussian_blur,
								bool is_tracking_enabled,
								bool is_detection_enabled) {

	if(sigma_of_gaussian_blur <= 2.0)
		return new Master(f_width, 
						  f_height, 
						  is_tracking_enabled, 
						  is_detection_enabled);
	else
		return new Master(f_width, 
						  f_height, 
						  is_tracking_enabled, 
						  is_detection_enabled,
						  sigma_of_gaussian_blur);
}

void destroyHETLDMasterModule(Master *master) {

	delete master;
}
