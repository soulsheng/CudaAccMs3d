#ifndef VERTEX_H__
#define VERTEX_H__

#include "Joint.h"
#include "cuda_runtime.h"
#include <map>
#include <vector>

enum Matrix_Sort_Mode {
	NO_SORT,			//	   ���������ڶ�������������
	SERIAL_SORT,	//	˳���������ڶ��������ͬ���󣬹���㲥����
	CROSS_SORT	//	�����������ڶ���������ھ��󣬹���ϲ�����
};// �����Ծ���idΪ����������ʽ

//��������---------------------------------------------------------
//typedef float4 Vertex; // ���꣺(x,y,z);�ؽ�������w
template<typename T>
struct Vertexes{

	// ��ȡ�������꣬�ڴ浽�Դ�
	void initialize(int size, float* pBufferCoord, int* pBufferIndex){
		nSize = size;
		pVertex = new T[nSize];
		for(int i=0;i<nSize;i++){
			pVertex[i].x = pBufferCoord[i*3];
			pVertex[i].y = pBufferCoord[i*3+1];
			pVertex[i].z = pBufferCoord[i*3+2];
			pVertex[i].w = pBufferIndex[i] * 1.0f;
		}

		cudaMalloc( &pVertexDevice, sizeof(T) * nSize ) ;//Vertex[nSize];
		cudaMemcpy( pVertexDevice, pVertex, sizeof(T) * nSize, cudaMemcpyHostToDevice );

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
		indexMatrix.clear();

		T* pVertexTemp = new T[nSize];
		for( i=0;i<nSize;i++){
			pVertexTemp[i] = pVertex[ pIndexMatrix[i] ];
		}

		memcpy( pVertex, pVertexTemp, sizeof(T) * nSize );
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

		// �����ռ�����  ���磺0,0,0...0,  1,1,1...1, 2,2,2...2 ,...
		for (i=0;i<nSizeJoint;i++)
		{
			subIndex = new std::vector<int>;
			indexMatrix.insert(std::make_pair(i,subIndex));
		}

		for( i=0;i<nSize;i++){
			int indexM = int(pVertex[i].w + 0.5);// float to int
			indexMatrix[indexM]->push_back(i);
		}

		// ����ѭ�� ���磺0,1,2,3,4...16, 0,1,2,3,4...16 ...
		i=0;
		int nLengthMax = 0;// �Ӷ��� ��󳤶�
		for( itr = indexMatrix.begin(); itr!=indexMatrix.end(); itr++){
			int size = itr->second->size();
			if(size > nLengthMax) nLengthMax=size ;
		}

		// �����Ӷ���
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

		// ����ѭ�������������洢
		T* pVertexTemp = new T[nSize];
		for( i=0;i<nSize;i++){
			pVertexTemp[i] = pVertex[ pIndexMatrix[i] ];
		}
		memcpy( pVertex, pVertexTemp, sizeof(T) * nSize );
		
		// �ͷ���ʱ�ڴ�ռ�
		delete[] pVertexTemp;
		delete[] pIndexMatrix;
		
		for ( itr=indexMatrix.begin();itr!=indexMatrix.end();itr++)
		{
			subIndex = itr->second;
			delete subIndex;
		}
	}

	//  �����ڴ�
	void initialize(int size, int sizeJoint, bool bDevice = true){
		nSize = size;
		nSizeJoint = sizeJoint;

		pVertex = new T[nSize];
		
		if( bDevice ){
		cudaMalloc( &pVertexDevice, sizeof(T) * nSize ) ;//Vertex[nSize];
		}
		else{
			pVertexDevice = NULL;
		}
	}

	//  ���ó�ʼֵ
	void setDefault(Matrix_Sort_Mode mode, bool bDevice = true)
	{
		eSort = mode;
		for(int i=0;i<nSize;i++){
			pVertex[i].x = rand() * 1.0f;
			pVertex[i].y = rand() * 1.0f;
			pVertex[i].z = rand() * 1.0f;
			pVertex[i].w = rand() % nSizeJoint  * 1.0f;
		}

		switch( eSort )
		{
		case NO_SORT:
			break;

		case SERIAL_SORT:
			sort();
			break;		

		case CROSS_SORT:
			sortLoop();
			break;		
		}

		if( bDevice ){
			cudaMemcpy( pVertexDevice, pVertex, sizeof(T) * nSize, cudaMemcpyHostToDevice );
		}
		
	}

	// �ͷſռ�
	void unInitialize()
	{
		if (pVertex) delete[] pVertex;
		if (pVertexDevice) cudaFree(pVertex) ;
	}
	
	void copy( Vertexes& ref )
	{
		if( pVertex!=NULL && ref.pVertex!=NULL )
			memcpy( pVertex, ref.pVertex, sizeof(T) * nSize );
		if( pVertexDevice!=NULL )
			cudaMemcpy( pVertexDevice, pVertex, sizeof(T) * nSize, cudaMemcpyHostToDevice );
	}

	T*  pVertex, *pVertexDevice;
	int   nSize;// �������Ŀ
	int   nSizeJoint;// �ؽڵ���Ŀ
	Matrix_Sort_Mode		eSort;
};// ����ļ���

#endif//VERTEX_H__