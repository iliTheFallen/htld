/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/

/*
 * fbtrack.h
 *
 *  Created on: 29.04.2011
 *      Author: Georg Nebehay
 */

#ifndef FBTRACK_H_
#define FBTRACK_H_

#include <opencv/cv.h>

#include "TLDGlobals.h"//Declare First for Compiler Options...
#ifdef USE_HTLD
	#include "hetld_api.hpp"
#endif

/*
 * @param imgI       Image contain Object with known BoundingBox
 * @param imgJ       Following Image.
 * @param bb         Bounding box of object to track in imgI.
 *                   Format x1,y1,x2,y2
 * @param scaleshift returns relative scale change of bb
 */
#ifndef USE_HTLD
	int fbtrack(IplImage *imgI, IplImage *imgJ, float *bb, float *bbnew, float *scaleshift);
#else
	int fbtrack(IplImage *imgI, 
		        IplImage *imgJ, 
				float *bb, 
				float *bbnew, 
				float *scaleshift,
				FastTracking *fastTr,
				FAST_TRACKING_STR *fastTrStr);
#endif

#endif /* FBTRACK_H_ */
