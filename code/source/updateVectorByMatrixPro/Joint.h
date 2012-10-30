#ifndef JOINT_H__
#define JOINT_H__

#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"

//#define		ALIGNED_STRUCT		1// ���뿪�أ�0�����룬1����
#define		USE_SHARED			0// �����أ�0������1����
#define		SEPERATE_STRUCT	0// �ṹ���ֿ��أ�0����֣�1���
#define		USE_MEMORY_BUY_TIME		1	// �Կռ任ʱ�䣬 1��ʾ����0��ʾ��������bug��

#define		SEPERATE_STRUCT_FULLY		0 // �ṹ�峹�ײ�ֿ��أ�0����֣�1���

#define    JOINT_SIZE    100
#define    MATRIX_SIZE_LINE    4//3
#define    JOINT_WIDTH    ((MATRIX_SIZE_LINE)*4)//12


struct Vector4 { float x,y,z,w; };


#if SEPERATE_STRUCT

#if !SEPERATE_STRUCT_FULLY
//typedef Vector4*  Matrix[MATRIX_SIZE_LINE];// ����


struct Joints{

	// ��ȡ�ؽھ���
	void initialize( int size, float* pBufferMatrix ){
		
	}

	// ��ȡ�ؽھ��� ģ��
	void initialize( int size ){
		nSize = size;
		pMatrix = new Vector4[nSize*MATRIX_SIZE_LINE];
		for(int i=0;i<MATRIX_SIZE_LINE;i++){
			for(int j=0;j<nSize;j++){
				int index = i * nSize + j;
				pMatrix[index].x = rand() % nSize / (nSize * 1.0f);
				pMatrix[index].y = rand() % nSize / (nSize * 1.0f);
				pMatrix[index].z = rand() % nSize / (nSize * 1.0f);
				if(i<3)	pMatrix[index].w = 0.0f;
				else		pMatrix[index].w = 1.0f;
				}
			}
		cudaMalloc( &pMatrixDevice, sizeof(Vector4) * nSize*MATRIX_SIZE_LINE ) ;

		pMatrixPrevious = new Vector4[nSize*MATRIX_SIZE_LINE];
		for(int i=0;i<MATRIX_SIZE_LINE;i++){
			for(int j=0;j<nSize;j++){
				int index = i * nSize + j;
				pMatrixPrevious[index].x = rand() % nSize / (nSize * 1.0f);
				pMatrixPrevious[index].y = rand() % nSize / (nSize * 1.0f);
				pMatrixPrevious[index].z = rand() % nSize / (nSize * 1.0f);
				if(i<3)	pMatrixPrevious[index].w = 0.0f;
				else		pMatrixPrevious[index].w = 1.0f;
				}
			}
		cudaMalloc( &pMatrixDevicePrevious, sizeof(Vector4) * nSize*MATRIX_SIZE_LINE ) ;
	}


	// �ͷſռ�
	void unInitialize()
	{
			if (pMatrix) delete[] pMatrix;
			if (pMatrixDevice) cudaFree(pMatrixDevice) ;
			if (pMatrixPrevious) delete[] pMatrixPrevious;
			if (pMatrixDevicePrevious) cudaFree(pMatrixDevicePrevious) ;
	}

	Vector4*  pMatrix, *pMatrixDevice;
	Vector4*  pMatrixPrevious, *pMatrixDevicePrevious; // �ؽھ��� ��һ֡
	int   nSize;// �ؽڵ���Ŀ

};// �ؽڵļ���

#else // SEPERATE_STRUCT_FULLY

//typedef float*  Matrix[JOINT_WIDTH];// ����


struct Joints{

	// ��ȡ�ؽھ���
	void initialize( int size, float* pBufferMatrix ){
		nSize = size;
		pMatrix = new float[nSize*JOINT_WIDTH];
		memcpy( pMatrix, pBufferMatrix, nSize*JOINT_WIDTH*sizeof(float) );
		
	}

	// ��ȡ�ؽھ��� ģ��
	void initialize( int size ){
		nSize = size;
		pMatrix = new float[nSize*JOINT_WIDTH];
		for(int i=0;i<nSize*JOINT_WIDTH;i++){
			pMatrix[i] = rand() % nSize / (nSize * 1.0f);
		}
		// ���һ��0,0,0,1
		for(int i=0;i<nSize*JOINT_WIDTH;i++){
			if( (i/nSize+1)%4 == 0)			pMatrix[i] = 0.0f;
			if( (i/nSize+1)%16 == 0)		pMatrix[i] = 1.0f;
		}

		cudaMalloc( &pMatrixDevice, sizeof(float) * nSize * JOINT_WIDTH) ;

		pMatrixPrevious = new float[nSize*JOINT_WIDTH];
		for(int i=0;i<nSize*JOINT_WIDTH;i++){
			pMatrixPrevious[i] = rand() % nSize / (nSize * 1.0f);
		}
		// ���һ��0,0,0,1
		for(int i=0;i<nSize*JOINT_WIDTH;i++){
			if( (i/nSize+1)%4 == 0 )		pMatrixPrevious[i] = 0.0f;
			if( (i/nSize+1)%16 == 0 )	pMatrixPrevious[i] = 1.0f;
		}

		cudaMalloc( &pMatrixDevicePrevious, sizeof(float) * nSize * JOINT_WIDTH) ;
	}

	// �ͷſռ�
	void unInitialize()
	{
		if (pMatrix) delete[] pMatrix;	
		if (pMatrixDevice) cudaFree(pMatrixDevice) ;
		
		if (pMatrixPrevious) delete[] pMatrixPrevious;
		if (pMatrixDevicePrevious) cudaFree(pMatrixDevicePrevious) ;
	}

	float		*pMatrix, *pMatrixDevice;
	float		*pMatrixPrevious, *pMatrixDevicePrevious; // �ؽھ��� ��һ֡
	int   nSize;// �ؽڵ���Ŀ

};// �ؽڵļ���

#endif // SEPERATE_STRUCT_FULLY

#else // SEPERATE_STRUCT

//�ؽھ���---------------------------------------------------------
//typedef Vector4  Matrix[MATRIX_SIZE_LINE];// ����
template<typename T>
struct Joints{
	//typedef T Matrix[MATRIX_SIZE_LINE];// ����
	// ��ȡ�ؽھ���
	void initialize( int size, float* pBufferMatrix ){
		
	}

	// ��ȡ�ؽھ��� ģ��
	void initialize( int size ){
		nSize = size;
		pMatrix = new T[nSize*MATRIX_SIZE_LINE];
		for(int i=0;i<nSize;i++){
			for(int j=0;j<MATRIX_SIZE_LINE;j++){
				pMatrix[ i*MATRIX_SIZE_LINE + j ].x = rand() % nSize / (nSize * 1.0f);
				pMatrix[ i*MATRIX_SIZE_LINE + j ].y = rand() % nSize / (nSize * 1.0f);
				pMatrix[ i*MATRIX_SIZE_LINE + j ].z = rand() % nSize / (nSize * 1.0f);
				if(j<3)	pMatrix[ i*MATRIX_SIZE_LINE + j ].w = 0.0f;
				else		pMatrix[ i*MATRIX_SIZE_LINE + j ].w = 1.0f;
			}
		}

		pMatrixPrevious = new T[nSize*MATRIX_SIZE_LINE];
		for(int i=0;i<nSize;i++){
			for(int j=0;j<MATRIX_SIZE_LINE;j++){
				pMatrixPrevious[ i*MATRIX_SIZE_LINE + j ].x = rand() % nSize / (nSize * 1.0f);
				pMatrixPrevious[ i*MATRIX_SIZE_LINE + j ].y = rand() % nSize / (nSize * 1.0f);
				pMatrixPrevious[ i*MATRIX_SIZE_LINE + j ].z = rand() % nSize / (nSize * 1.0f);
				if(j<3)	pMatrixPrevious[ i*MATRIX_SIZE_LINE + j ].w = 0.0f;
				else		pMatrixPrevious[ i*MATRIX_SIZE_LINE + j ].w = 1.0f;
			}
		}

		cudaMalloc( &pMatrixDevice, sizeof(T) * nSize * MATRIX_SIZE_LINE ) ;
		cudaMalloc( &pMatrixDevicePrevious, sizeof(T) * nSize * MATRIX_SIZE_LINE ) ;
	}

	// �ͷſռ�
	void unInitialize()
	{
		if (pMatrix) delete[] pMatrix;
		if (pMatrixPrevious) delete[] pMatrixPrevious;
		if (pMatrixDevice) cudaFree(pMatrixDevice) ;
		if (pMatrixDevicePrevious) cudaFree(pMatrixDevicePrevious) ;
	}

	T*  pMatrix, *pMatrixDevice;
	T*  pMatrixPrevious, *pMatrixDevicePrevious; // �ؽھ��� ��һ֡
	int   nSize;// �ؽڵ���Ŀ

};// �ؽڵļ���

#endif

#endif//JOINT_H__


