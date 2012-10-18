#pragma once

#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"

#define		ALIGNED_STRUCT		1// ���뿪�أ�0�����룬1����
#define		USE_SHARED			0// �����أ�0������1����

#if ALIGNED_STRUCT
typedef float4	Vector4;
typedef Vector4*  Matrix[3];// ����


struct Joints{

	// ��ȡ�ؽھ���
	void initialize( int size, float* pBufferMatrix ){
		nSize = size;
		for (int i=0;i<3;i++)
		{
			pMatrix[i] = new Vector4[nSize];
			int offset = sizeof(Vector4)/sizeof(float) * nSize;
			memcpy( pMatrix[i], pBufferMatrix + offset * i, offset*sizeof(float) );
			cudaMalloc( &pMatrixDevice[i], offset*sizeof(float) ) ;
		}
	}

	// ��ȡ�ؽھ��� ģ��
	void initialize( int size ){
		nSize = size;
		for(int i=0;i<3;i++){
			pMatrix[i] = new Vector4[nSize];
			for(int j=0;j<nSize;j++){
				pMatrix[i][j].x = rand() * 1.0f;
				pMatrix[i][j].y = rand() * 1.0f;
				pMatrix[i][j].z = rand() * 1.0f;
				pMatrix[i][j].w = rand() * 1.0f;
			}
			cudaMalloc( &pMatrixDevice[i], sizeof(Vector4) * nSize ) ;
		}
	}

	// �ͷſռ�
	void unInitialize()
	{
		for(int i=0;i<3;i++){
			if (pMatrix[i]) delete[] pMatrix[i];
			if (pMatrixDevice[i]) cudaFree(pMatrixDevice[i]) ;
		}
	}

	Matrix  pMatrix, pMatrixDevice;
	int   nSize;// �ؽڵ���Ŀ

};// �ؽڵļ���

#else
struct Vector4 { float x,y,z,w; };

//�ؽھ���---------------------------------------------------------
typedef Vector4  Matrix[3];// ����

struct Joints{

	// ��ȡ�ؽھ���
	void initialize( int size, float* pBufferMatrix ){
		nSize = size;
		pMatrix = new Matrix[nSize];
		memcpy( pMatrix, pBufferMatrix, sizeof(Matrix) * nSize );

		cudaMalloc( &pMatrixDevice, sizeof(Matrix) * nSize ) ;
	}

	// ��ȡ�ؽھ��� ģ��
	void initialize( int size ){
		nSize = size;
		pMatrix = new Matrix[nSize];
		for(int i=0;i<nSize;i++){
			for(int j=0;j<3;j++){
				pMatrix[i][j].x = rand() * 1.0f;
				pMatrix[i][j].y = rand() * 1.0f;
				pMatrix[i][j].z = rand() * 1.0f;
				pMatrix[i][j].w = rand() * 1.0f;
			}
		}

		cudaMalloc( &pMatrixDevice, sizeof(Matrix) * nSize ) ;
	}

	// �ͷſռ�
	void unInitialize()
	{
		if (pMatrix) delete[] pMatrix;
		if (pMatrixDevice) cudaFree(pMatrixDevice) ;
	}

	Matrix*  pMatrix, *pMatrixDevice;
	int   nSize;// �ؽڵ���Ŀ

};// �ؽڵļ���

#endif


