#pragma once

#include <stdlib.h>
#include <string.h>
#include "Vector.h"

#define    MATRIX_SIZE_LINE    3//3

//关节矩阵---------------------------------------------------------
//typedef float4  Matrix[3];// 矩阵

struct Joints{

	// 获取关节矩阵
	void initialize( int size, float* pBufferMatrix ){
		nSize = size;
		pMatrix = new float[nSize*MATRIX_SIZE_LINE*4];
		memcpy( pMatrix, pBufferMatrix, sizeof(float) * nSize*MATRIX_SIZE_LINE*4 );
	}

	// 获取关节矩阵 模拟
	void initialize( int size ){
		nSize = size;
		//pMatrix = new float[nSize*MATRIX_SIZE_LINE*4];
		pMatrix = (float*) _aligned_malloc(nSize*MATRIX_SIZE_LINE*4 * sizeof(float), 16);

		for(int i=0;i<nSize;i++){
			for(int j=0;j<MATRIX_SIZE_LINE;j++){
				for(int k=0;k<4;k++){
					pMatrix[i*MATRIX_SIZE_LINE*4 +4*j+k] = rand() * 1.0f;
				}
			}
		}
	}

	// 释放空间
	void unInitialize()
	{
		if (pMatrix)  _aligned_free(pMatrix);
	}

	float*  pMatrix;
	int   nSize;// 关节的数目

};// 关节的集合

