#pragma once

#include "Joint.h"
#include "Vector.h"

//��������---------------------------------------------------------
//typedef float4 Vertex; // ���꣺(x,y,z);�ؽ�������w
#define    VERTEX_VECTOR_SIZE    4

struct Vertexes{

	// ��ȡ�������� ģ��
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

	// �ͷſռ�
	void unInitialize()
	{
		if (pVertex) delete[] pVertex;
		if (pIndex) delete[] pIndex;
	}

	float*  pVertex;
	int   nSize;// �������Ŀ
	int*		pIndex;
};// ����ļ���