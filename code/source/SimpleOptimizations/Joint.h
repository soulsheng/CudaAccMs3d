#pragma once

#include <stdlib.h>
#include <string.h>

#define    MATRIX_SIZE_LINE    3//3
#define    VECTOR_FLOAT4    1//3

//关节矩阵---------------------------------------------------------
//typedef float4  Matrix[3];// 矩阵
#if !VECTOR_FLOAT4
struct Joints{

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

#else

struct Joints{

	// 获取关节矩阵 模拟
	void initialize( int size ){
		nSize = size;
		//pMatrix = new float[nSize*MATRIX_SIZE_LINE*4];
		pMatrix = (cl_float4*) _aligned_malloc(nSize*MATRIX_SIZE_LINE * sizeof(cl_float4), 16);

		for(int i=0;i<nSize;i++){
			for(int j=0;j<MATRIX_SIZE_LINE;j++){
				for(int k=0;k<4;k++){
					pMatrix[i*MATRIX_SIZE_LINE +j].s[k] = rand() * 1.0f;
				}
			}
		}
	}

	// 释放空间
	void unInitialize()
	{
		if (pMatrix)  _aligned_free(pMatrix);
	}

	cl_float4*  pMatrix;
	int   nSize;// 关节的数目

};// 关节的集合

#endif