#pragma once

#include "Joint.h"
#include "cuda_runtime.h"
#include <map>
#include <vector>

#define		SORT_ARRAY			0// 顶点按矩阵索引排序：0不排序，1排序
#define		SORT_ARRAY_LOOP		0// 顶点按矩阵索引循环交替排序：0不排序，1排序    排序如：0~100,0~100...

//顶点坐标---------------------------------------------------------
//typedef float4 Vertex; // 坐标：(x,y,z);关节索引：w
struct Vertexes{

	// 获取顶点坐标，内存到显存
	void initialize(int size, float* pBufferCoord, int* pBufferIndex){
		nSize = size;
		pVertex = new Vector4[nSize];
		for(int i=0;i<nSize;i++){
			pVertex[i].x = pBufferCoord[i*3];
			pVertex[i].y = pBufferCoord[i*3+1];
			pVertex[i].z = pBufferCoord[i*3+2];
			pVertex[i].w = pBufferIndex[i] * 1.0f;
		}

		cudaMalloc( &pVertexDevice, sizeof(Vector4) * nSize ) ;//Vertex[nSize];
		cudaMemcpy( pVertexDevice, pVertex, sizeof(Vector4) * nSize, cudaMemcpyHostToDevice );

	}
	void sort()
	{
		int i=0;
		std::multimap<int, int> indexMatrix;
		for( i=0;i<nSize;i++){
			int indexM = int(pVertex[i].w + 0.5);// float to int
			indexMatrix.insert( std::make_pair(indexM, i) );
		}

		int *pIndexMatrix = new int[nSize];
		i=0;
		for(std::multimap<int, int>::iterator itr = indexMatrix.begin(); itr!=indexMatrix.end(); itr++, i++){
			pIndexMatrix[i] = itr->second;
		}

		Vector4* pVertexTemp = new Vector4[nSize];
		for( i=0;i<nSize;i++){
			pVertexTemp[i] = pVertex[ pIndexMatrix[i] ];
		}

		memcpy( pVertex, pVertexTemp, sizeof(float4) * nSize );
		delete[] pVertexTemp;
		delete[] pIndexMatrix;
	}

	void sortLoop()
	{
		typedef std::vector<int> SubIndex;
		typedef std::vector<int>::iterator SubIndexItr;
		typedef std::map<int, SubIndex*> IndexMatrix;
		typedef std::map<int, SubIndex*>::iterator IndexMatrixItr;

		int i=0;
		IndexMatrix		indexMatrix;
		IndexMatrixItr	itr;
		SubIndex		*subIndex;

		// 按组收集索引  形如：0,0,0...0,  1,1,1...1, 2,2,2...2 ,...
		for (i=0;i<nSizeJoint;i++)
		{
			subIndex = new std::vector<int>;
			indexMatrix.insert(std::make_pair(i,subIndex));
		}

		for( i=0;i<nSize;i++){
			int indexM = int(pVertex[i].w + 0.5);// float to int
			indexMatrix[indexM]->push_back(i);
		}

		// 构造循环 形如：0,1,2,3,4...16, 0,1,2,3,4...16 ...
		i=0;
		int nLengthMax = 0;// 子队列 最大长度
		for( itr = indexMatrix.begin(); itr!=indexMatrix.end(); itr++){
			int size = itr->second->size();
			if(size > nLengthMax) nLengthMax=size ;
		}

		// 遍历子队列
		int *pIndexMatrix = new int[nSize];
		int j=0;
		for (i=0;i<nLengthMax;i++)
		{
			for( itr = indexMatrix.begin(); itr!=indexMatrix.end(); itr++){
				subIndex = itr->second;
				if ( i<subIndex->size() )
				{
					pIndexMatrix[j++] = (*subIndex)[i];
				}
			}

		}

		// 按照循环索引，调整存储
		Vector4* pVertexTemp = new Vector4[nSize];
		for( i=0;i<nSize;i++){
			pVertexTemp[i] = pVertex[ pIndexMatrix[i] ];
		}
		memcpy( pVertex, pVertexTemp, sizeof(float4) * nSize );
		
		// 释放临时内存空间
		delete[] pVertexTemp;
		delete[] pIndexMatrix;
		
		for ( itr=indexMatrix.begin();itr!=indexMatrix.end();itr++)
		{
			subIndex = itr->second;
			delete subIndex;
		}
	}

	// 获取顶点坐标 模拟
	void initialize(int size, int sizeJoint){
		nSize = size;
		pVertex = new Vector4[nSize];
		for(int i=0;i<nSize;i++){
			pVertex[i].x = rand() * 1.0f;
			pVertex[i].y = rand() * 1.0f;
			pVertex[i].z = rand() * 1.0f;
			pVertex[i].w = rand() % sizeJoint  * 1.0f;
		}
#if SORT_ARRAY
		sort();
#endif
		
		nSizeJoint = sizeJoint;
#if SORT_ARRAY_LOOP
		sortLoop();
#endif

		cudaMalloc( &pVertexDevice, sizeof(Vector4) * nSize ) ;//Vertex[nSize];
		cudaMemcpy( pVertexDevice, pVertex, sizeof(Vector4) * nSize, cudaMemcpyHostToDevice );

	}

	// 释放空间
	void unInitialize()
	{
		if (pVertex) delete[] pVertex;
		if (pVertexDevice) cudaFree(pVertex) ;
	}

	Vector4*  pVertex, *pVertexDevice;
	int   nSize;// 顶点的数目
	int   nSizeJoint;// 关节的数目

};// 顶点的集合