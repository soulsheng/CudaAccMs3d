#pragma once

#include "Joint.h"
#include "cuda_runtime.h"

//顶点坐标---------------------------------------------------------
typedef float4 Vertex; // 坐标：(x,y,z);关节索引：w

struct Vertexes{

	// 获取顶点坐标，内存到显存
	void initialize(int size, float* pBufferCoord, int* pBufferIndex){
		nSize = size;
		pVertex = new Vertex[nSize];
		for(int i=0;i<nSize;i++){
			pVertex[i].x = pBufferCoord[i*3];
			pVertex[i].y = pBufferCoord[i*3+1];
			pVertex[i].z = pBufferCoord[i*3+2];
			pVertex[i].w = pBufferIndex[i] * 1.0f;
		}

		cudaMalloc( &pVertexDevice, sizeof(Vertex) * nSize ) ;//Vertex[nSize];
		cudaMemcpy( pVertexDevice, pVertex, sizeof(Vertex) * nSize, cudaMemcpyHostToDevice );

	}

	// 获取顶点坐标 模拟
	void initialize(int size, int sizeJoint){
		nSize = size;
		pVertex = new Vertex[nSize];
		for(int i=0;i<nSize;i++){
			pVertex[i].x = rand() * 1.0f;
			pVertex[i].y = rand() * 1.0f;
			pVertex[i].z = rand() * 1.0f;
			pVertex[i].w = rand() % sizeJoint  * 1.0f;
		}

		cudaMalloc( &pVertexDevice, sizeof(Vertex) * nSize ) ;//Vertex[nSize];
		cudaMemcpy( pVertexDevice, pVertex, sizeof(Vertex) * nSize, cudaMemcpyHostToDevice );

	}

	// 释放空间
	void unInitialize()
	{
		if (pVertex) delete[] pVertex;
		if (pVertexDevice) cudaFree(pVertex) ;
	}

	Vertex*  pVertex, *pVertexDevice;
	int   nSize;// 顶点的数目

};// 顶点的集合