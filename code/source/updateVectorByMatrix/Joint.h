#pragma once

#include <stdlib.h>
#include <string.h>
#include "Vector.h"

#define    MATRIX_SIZE_LINE    3//3

//�ؽھ���---------------------------------------------------------
//typedef float4  Matrix[3];// ����

struct Joints{

	// ��ȡ�ؽھ���
	void initialize( int size, float* pBufferMatrix ){
		nSize = size;
		pMatrix = new float4[nSize*MATRIX_SIZE_LINE];
		memcpy( pMatrix, pBufferMatrix, sizeof(float4) * nSize*MATRIX_SIZE_LINE );
	}

	// ��ȡ�ؽھ��� ģ��
	void initialize( int size ){
		nSize = size;
		pMatrix = new float4[nSize*MATRIX_SIZE_LINE];
		for(int i=0;i<nSize;i++){
			for(int j=0;j<MATRIX_SIZE_LINE;j++){
				pMatrix[i*MATRIX_SIZE_LINE+j].x = rand() * 1.0f;
				pMatrix[i*MATRIX_SIZE_LINE+j].y = rand() * 1.0f;
				pMatrix[i*MATRIX_SIZE_LINE+j].z = rand() * 1.0f;
				pMatrix[i*MATRIX_SIZE_LINE+j].w = rand() * 1.0f;
			}
		}
	}

	// �ͷſռ�
	void unInitialize()
	{
		if (pMatrix) delete[] pMatrix;
	}

	float4*  pMatrix;
	int   nSize;// �ؽڵ���Ŀ

};// �ؽڵļ���

