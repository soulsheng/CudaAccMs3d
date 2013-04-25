#pragma once

#include "Joint.h"
#include "Vector.h"

//顶点坐标---------------------------------------------------------
//typedef float4 Vertex; // 坐标：(x,y,z);关节索引：w
#define    VERTEX_VECTOR_SIZE    4

struct Vertexes{

	// 获取顶点坐标 模拟
	void initialize(int size, int sizeJoint){
		nSize = size;
		pVertex = new float[nSize*VERTEX_VECTOR_SIZE];
		pIndex = new int[nSize];
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
		if (pVertex) delete[] pVertex;
		if (pIndex) delete[] pIndex;
	}

	float*  pVertex;
	int   nSize;// 顶点的数目
	int*		pIndex;
};// 顶点的集合