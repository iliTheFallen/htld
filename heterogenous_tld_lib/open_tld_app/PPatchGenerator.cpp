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
* PPatchGenerator.cpp
* 
* Author: Ilker GURCAN
*/

#include<iostream>
#include<climits>
#include<cmath>

#include "PPatchGenerator.h"
#include "hetld_macros.hpp"

void bbHull(int *windowOffsets, 
			std::vector<int> &indices, 
			int imW, 
			int *hull) {

	int *bb;
	int rMin, cMin, rMax, cMax;
	std::vector<int>::iterator it;
	int minX = INT_MAX;
	int minY = INT_MAX;
	int maxX = 0;
	int maxY = 0;

	if(indices.size() < 1) {
		hull[0] = 0;
		hull[1] = 0;
		hull[2] = 0;
		hull[3] = 0;
		return;
	}

	it = indices.begin();
	while(it != indices.end()) {
		bb   = windowOffsets + NUM_OF_BBOX_ATTRS_ON_CPU * (*it);
		rMin = (int)floorf(bb[TOP_LEFT_CORNER] / (float)imW);
		cMin =  bb[TOP_LEFT_CORNER] - rMin * imW;
		rMax = (int)floorf(bb[BOTTOM_RIGHT_CORNER] / (float)imW);
		cMax =  bb[BOTTOM_RIGHT_CORNER] - rMax * imW;
		//Min Point...
		if(minX > cMin)
			minX = cMin;
		if(minY > rMin)
			minY = rMin;
		//Max Point...
		if(maxX < cMax)
			maxX = cMax;
		if(maxY < rMax)
			maxY = rMax;
		it++;
	}

	hull[0] = minX;
	hull[1] = minY;
	hull[2] = maxX;
	hull[3] = maxY;
}

/**
*@param img Blurred Image
*@param h Transformation Matrix(3x3)
*@param bb Region of Interest (xMin, yMin, xMax, yMax)
*@param result Image Transformed by h
**/
void warpImageROI(cv::Mat &img, 
				  double xMin,
				  double yMin,
				  double xMax,
				  double yMax,
				  int bbW,
				  int bbH,
				  cv::Mat &h,
				  unsigned char filledColor,
				  unsigned char *result) {

	double curx, cury, curz, wx, wy, wz, ox, oy, oz;
	int x, y;
	double xx, yy;
	int i;
	unsigned char *tmp;
	unsigned char *rowY, *rowYP1;

	cv::Mat invT = h.inv(cv::DECOMP_LU);//Matlab's inv Method Only Takes the Inverse of Square, non-Singular Matrices...
	double *invPtr = invT.ptr<double>();
	ox = *(invPtr + 2);
	oy = *(invPtr + invT.cols + 2);
	oz = *(invPtr + 2 * invT.cols + 2);

	yy = yMin;
	for(int j=0; j<bbH; j++) {
		/* calculate x, y for current row */
		curx = *(invPtr + 1) * yy + ox;
		cury = *(invPtr + invT.cols + 1) * yy + oy;
		curz = *(invPtr + 2 * invT.cols + 1) * yy + oz;
		xx   = xMin; 

		for(i=0; i<bbW; i++) {
			/* calculate x, y in current column */
			wx = *(invPtr)*xx + curx;
			wy = *(invPtr + invT.cols) * xx + cury;
			wz = *(invPtr + 2 * invT.cols) * xx + curz;
			wx /= wz; wy /= wz;
			
			x = (int)floor(wx);
			y = (int)floor(wy);

			if(x>=0 && y>=0) {
				wx -= x; wy -= y;//[0,1]...
				if(x+1 == img.cols && wx == 1.0)
					x--;
				if(y+1 == img.rows && wy == 1.0)
					y--;
				if((x+1) < img.cols && (y+1) < img.rows) {
					//Different Pixels Nearby Have Different Contributions to the Result!...
					/* img[x,y]*(1-wx)*(1-wy) + 
						img[x+1,y]*wx*(1-wy) +
						img[x,y+1]*(1-wx)*wy + 
						img[x+1,y+1]*wx*wy 
						*/
					rowY   = img.ptr<unsigned char>(y);//In order to Have Access to Elements Faster...
					rowYP1 = img.ptr<unsigned char>(y + 1);//In order to Have Access to Elements Faster...
					*result++ = cv::saturate_cast<unsigned char>(
						            (rowY[x]   * (1.0 - wx) + rowY[x + 1]   * wx) * (1.0 - wy) + 
									(rowYP1[x] * (1.0 - wx) + rowYP1[x + 1] * wx) * wy
								);
				} else 
					*result++ = filledColor;
			} else 
				*result++ = filledColor;
			xx = xx + 1;
		}//End of Inner-for-Loop...
		yy = yy + 1;
	}//End of Outermost-for-Loop...
}

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
				  cv::Mat &result) {

	int cpX         = (hull[0] + hull[2]) / 2;
	int cpY         = (hull[1] + hull[3]) / 2;
	cv::Mat h       = cv::Mat::eye(3, 3, CV_64FC1);
	cv::Mat temp    = cv::Mat::eye(3, 3, CV_64FC1);
	double *tempPtr = temp.ptr<double>();
	double sc;
	double ang, ca, sa;
	double shR, shC;
	double xMin, yMin, xMax, yMax;
	unsigned char *resPtr;
	double *noisePtr;
	int size;

	
	//****Translating...
	shR  = shift * bbH * (rng.uniform(1e-4, 1.)-0.5);
	shC  = shift * bbW * (rng.uniform(1e-4, 1.)-0.5);
	*(tempPtr + 2)             = shC;
	*(tempPtr + temp.cols + 2) = shR;
	h                          *= temp;
	//Reset...
	*(tempPtr + 2)              = 0.0;
	*(tempPtr +  temp.cols + 2) = 0.0;

	//****Rotating...
	ang  = 2 * CV_PI / 360.0 * angle * (rng.uniform(1e-4, 1.)-0.5);
	ca   = cos(ang);
    sa   = sin(ang);
	*tempPtr                   = ca;
	*(tempPtr + 1)             = -sa;
	*(tempPtr + temp.cols)     = sa;
	*(tempPtr + temp.cols + 1) = ca;
	h                          *= temp;
	//Reset...
	*tempPtr                   = 1.0;
	*(tempPtr + 1)             = 0.0;
	*(tempPtr + temp.cols)     = 0.0;
	*(tempPtr + temp.cols + 1) = 1.0;
	
	//****Scaling...
	sc   = 1.0 - scale*(rng.uniform(1e-4, 1.)-0.5);
	*tempPtr                   = (double)sc;
	*(tempPtr + temp.cols + 1) = (double)sc;
	h                          *= temp;
	//Reset...
	*tempPtr                   = 1.0;
	*(tempPtr + temp.cols + 1) = 1.0;
	
	//****Shifting Center of BB to (0, 0)...
	*(tempPtr + 2)             = -cpX;
	*(tempPtr + temp.cols + 2) = -cpY;
	h                          *= temp;
	
	//Now Warp the Patch...
	bbW--; 
	bbH--;
	xMin = -bbW / 2.0;
	yMin = -bbH / 2.0;
	xMax = bbW / 2.0;
	yMax = bbH / 2.0;
	warpImageROI(img,
				 xMin,
				 yMin,
				 xMax,
				 yMax,
				 bbW,
				 bbH,
				 h,
				 filledColor,
				 result.data);

	//Add Random Noise...
	rng.fill(noiseM, 
		     cv::RNG::NORMAL, 
			 cv::Mat::zeros(1,1,CV_64FC1), 
			 cv::Mat::ones(1,1,CV_64FC1));
	noiseM *= noise;
	//Here OpenCV Applies Saturation Arithmetic by Itself...
	cv::add(result, noise, result, cv::noArray(), CV_8UC1);
}
