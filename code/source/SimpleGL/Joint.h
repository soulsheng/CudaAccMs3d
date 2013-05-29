#pragma once

#include <stdlib.h>
#include <string.h>

#include "definition.h"

//�ؽھ���---------------------------------------------------------
//typedef float4  Matrix[3];// ����
#if !VECTOR_FLOAT4
struct Joints{

	// ��ȡ�ؽھ��� ģ��
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

	// �ͷſռ�
	void unInitialize()
	{
		if (pMatrix)  _aligned_free(pMatrix);
	}

	float*  pMatrix;
	int   nSize;// �ؽڵ���Ŀ

};// �ؽڵļ���

#else

template<typename F>
struct Joints{

	// ��ȡ�ؽھ��� ģ��
	void initialize( int size ){
		nSize = size;
		//pMatrix = new float[nSize*MATRIX_SIZE_LINE*4];
		pMatrix = (F*) _aligned_malloc(nSize*MATRIX_SIZE_LINE * sizeof(float)*4, 16);

		for(int i=0;i<nSize;i++){
			for(int j=0;j<MATRIX_SIZE_LINE;j++){
				for(int k=0;k<4;k++){
					pMatrix[ (i*MATRIX_SIZE_LINE +j)*4 + k] = rand() % nSize / (nSize * 1.0f);
				}
			}
		}
	}

	// �ͷſռ�
	void unInitialize()
	{
		if (pMatrix)  _aligned_free(pMatrix);
	}

	F*  pMatrix;
	int   nSize;// �ؽڵ���Ŀ

};// �ؽڵļ���

#endif