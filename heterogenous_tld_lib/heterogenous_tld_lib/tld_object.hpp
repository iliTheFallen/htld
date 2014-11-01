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

#pragma once

#include "global_declarations.hpp"
#include "fast_tracking.cuh"
#include "fast_detection.cuh"

class HETLD_API TLDObject
{
public:
	TLDObject(bool is_tracking_enabled,
			  bool is_detection_enabled);
	virtual ~TLDObject(void);
	int getObjectId() {return _object_id;};
	FAST_TRACKING_STR* getFastTrStr() {return _fast_tr_str;};
	FAST_DETECTION_STR* getFastDetStr() {return _fast_det_str;};
	void printFTS();
	void printFDS();
private:
	static int          _obj_seq;
	int                 _object_id;
	FAST_TRACKING_STR  *_fast_tr_str;
	FAST_DETECTION_STR *_fast_det_str;
};
