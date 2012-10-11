#pragma once

#include <stdlib.h>
#include <string.h>
#include "Vector.h"

//�ؽھ���---------------------------------------------------------
typedef float3  Matrix[3];// ����

struct Joints{

	// ��ȡ�ؽھ���
	void initialize( int size, float* pBufferMatrix ){
		nSize = size;
		pMatrix = new Matrix[nSize];
		memcpy( pMatrix, pBufferMatrix, sizeof(Matrix) * nSize );
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
			}
		}
	}

	Matrix*  pMatrix;
	int   nSize;// �ؽڵ���Ŀ

};// �ؽڵļ���

