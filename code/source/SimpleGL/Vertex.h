#pragma once

#include "Joint.h"
#if !VECTOR_FLOAT4
//顶点坐标---------------------------------------------------------
//typedef float4 Vertex; // 坐标：(x,y,z);关节索引：w
#define    VERTEX_VECTOR_SIZE    4

struct Vertexes{

	// 获取顶点坐标 模拟
	void initialize(int size, int sizeJoint){
		nSize = size;
		//pVertex = new float[nSize*VERTEX_VECTOR_SIZE];
		//pIndex = new int[nSize];
		pVertex = (float*) _aligned_malloc(nSize*VERTEX_VECTOR_SIZE * sizeof(float), 16);
		pIndex = (int*) _aligned_malloc(nSize * sizeof(int), 16);

		for(int i=0;i<nSize;i++){
			pVertex[i*VERTEX_VECTOR_SIZE + 0] = rand() * 1.0f;
			pVertex[i*VERTEX_VECTOR_SIZE + 1] = rand() * 1.0f;
			pVertex[i*VERTEX_VECTOR_SIZE + 2] = rand() * 1.0f;
			pVertex[i*VERTEX_VECTOR_SIZE + 3] = 1.0f;
			pIndex[i] = rand() % sizeJoint;
		}
	}

	// 释放空间
	void unInitialize()
	{
		if (pVertex) _aligned_free(pVertex);
		if (pIndex) _aligned_free(pIndex);
	}

	float*  pVertex;
	int   nSize;// 顶点的数目
	int*		pIndex;
};// 顶点的集合

#else

#define    VERTEX_VECTOR_SIZE    4

template<typename F>
struct Vertexes{

	// 获取顶点坐标 模拟
	void initialize(int size, int sizeJoint){
		nSize = size;
		//pVertex = new float[nSize*VERTEX_VECTOR_SIZE];
		//pIndex = new int[nSize];
		pVertex = (F*) _aligned_malloc(nSize * sizeof(float)*4, 16);
		pIndex = (unsigned short*) _aligned_malloc(nSize * sizeof(unsigned short) * SIZE_PER_BONE, 16);
		pWeight = (float*) _aligned_malloc(nSize * sizeof(float) * SIZE_PER_BONE, 16);

		float* pFormat = (float*)pVertex;
		for(int i=0;i<nSize;i++){
			for(int j=0;j<4;j++){
				if ( j%3 )				{
					pVertex[i*4+j] = rand() * 1.0f;
				} 
				else				{
					pVertex[i*4+j] = 1.0f;
				}
			}
			
			for(int j=0;j<SIZE_PER_BONE;j++) {
				pIndex[i + j*nSize] = rand() % sizeJoint;
				pWeight[i + j*nSize] = rand() % sizeJoint / (sizeJoint*1.0f);
			}
		}
	}

	// 释放空间
	void unInitialize()
	{
		if (pVertex) _aligned_free(pVertex);
		if (pIndex) _aligned_free(pIndex);
		if (pWeight) _aligned_free(pWeight);
	}

	F*  pVertex;
	int   nSize;// 顶点的数目
	unsigned short*		pIndex;
	float*		pWeight;

};// 顶点的集合

#endif