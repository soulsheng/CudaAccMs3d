#pragma once

#include "Joint.h"

#define  SIZE_PER_BONE		3

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

struct Vertexes{

	// 获取顶点坐标 模拟
	void initialize(int size, int sizeJoint){
		nSize = size;
		//pVertex = new float[nSize*VERTEX_VECTOR_SIZE];
		//pIndex = new int[nSize];
		pVertex = (cl_float4*) _aligned_malloc(nSize * sizeof(cl_float4), 16);
		pIndex = (cl_float4*) _aligned_malloc(nSize * sizeof(cl_float4), 16);
		pWeight = (cl_float4*) _aligned_malloc(nSize * sizeof(cl_float4), 16);

		for(int i=0;i<nSize;i++){
			pVertex[i].s[0] = rand() * 1.0f;
			pVertex[i].s[1] = rand() * 1.0f;
			pVertex[i].s[2] = rand() * 1.0f;
			pVertex[i].s[3] = 1.0f;

			for(int j=0; j<4; j++){
				pIndex[i].s[j] = rand() % sizeJoint;
			}

			for(int j=0; j<4; j++){
				pWeight[i].s[j] = rand() % sizeJoint / (sizeJoint*2.0f);
			}
		}
	}

	// 释放空间
	void unInitialize()
	{
		if (pVertex) _aligned_free(pVertex);
		if (pIndex) _aligned_free(pIndex);
	}

	cl_float4*  pVertex;
	int   nSize;// 顶点的数目
	
	cl_float4*		pIndex;
	cl_float4*		pWeight;

};// 顶点的集合

#endif