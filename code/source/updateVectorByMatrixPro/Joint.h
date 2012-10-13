#pragma once

#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"

//�ؽھ���---------------------------------------------------------
typedef float4  Matrix[3];// ����

struct Joints{

	// ��ȡ�ؽھ���
	void initialize( int size, float* pBufferMatrix ){
		nSize = size;
		pMatrix = new Matrix[nSize];
		memcpy( pMatrix, pBufferMatrix, sizeof(Matrix) * nSize );

		cudaMalloc( &pMatrixDevice, sizeof(Matrix) * nSize ) ;
		cudaMemcpy( pMatrixDevice, pMatrix, sizeof(Matrix)*nSize, cudaMemcpyHostToDevice );
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
		cudaMemcpy( pMatrixDevice, pMatrix, sizeof(Matrix) * nSize, cudaMemcpyHostToDevice );
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

