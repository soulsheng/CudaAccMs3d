#ifndef JOINT_H__
#define JOINT_H__

#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"

#define		ALIGNED_STRUCT		1// 对齐开关：0不对齐，1对齐
#define		USE_SHARED			1// 共享开关：0不共享，1共享
#define		SEPERATE_STRUCT	0// 结构体拆分开关：0不拆分，1拆分
#define		USE_MEMORY_BUY_TIME		0	// 以空间换时间， 1表示换，0表示不换（有bug）

#define		SEPERATE_STRUCT_FULLY		0 // 结构体彻底拆分开关：0不拆分，1拆分

#define    JOINT_SIZE    100
#define    MATRIX_SIZE_LINE    4//3
#define    JOINT_WIDTH    ((MATRIX_SIZE_LINE)*4)//12

#if ALIGNED_STRUCT
typedef float4	Vector4;
#else
struct Vector4 { float x,y,z,w; };
#endif

#if SEPERATE_STRUCT

#if !SEPERATE_STRUCT_FULLY
//typedef Vector4*  Matrix[MATRIX_SIZE_LINE];// 矩阵


struct Joints{

	// 获取关节矩阵
	void initialize( int size, float* pBufferMatrix ){
		
	}

	// 获取关节矩阵 模拟
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


	// 释放空间
	void unInitialize()
	{
			if (pMatrix) delete[] pMatrix;
			if (pMatrixDevice) cudaFree(pMatrixDevice) ;
			if (pMatrixPrevious) delete[] pMatrixPrevious;
			if (pMatrixDevicePrevious) cudaFree(pMatrixDevicePrevious) ;
	}

	Vector4*  pMatrix, *pMatrixDevice;
	Vector4*  pMatrixPrevious, *pMatrixDevicePrevious; // 关节矩阵 上一帧
	int   nSize;// 关节的数目

};// 关节的集合

#else // SEPERATE_STRUCT_FULLY

//typedef float*  Matrix[JOINT_WIDTH];// 矩阵


struct Joints{

	// 获取关节矩阵
	void initialize( int size, float* pBufferMatrix ){
		nSize = size;
		pMatrix = new float[nSize*JOINT_WIDTH];
		memcpy( pMatrix, pBufferMatrix, nSize*JOINT_WIDTH*sizeof(float) );
		
	}

	// 获取关节矩阵 模拟
	void initialize( int size ){
		nSize = size;
		pMatrix = new float[nSize*JOINT_WIDTH];
		for(int i=0;i<nSize*JOINT_WIDTH;i++){
			pMatrix[i] = rand() % nSize / (nSize * 1.0f);
		}
		// 最后一列0,0,0,1
		for(int i=0;i<nSize*JOINT_WIDTH;i++){
			if( (i/nSize+1)%4 == 0)			pMatrix[i] = 0.0f;
			if( (i/nSize+1)%16 == 0)		pMatrix[i] = 1.0f;
		}

		cudaMalloc( &pMatrixDevice, sizeof(float) * nSize * JOINT_WIDTH) ;

		pMatrixPrevious = new float[nSize*JOINT_WIDTH];
		for(int i=0;i<nSize*JOINT_WIDTH;i++){
			pMatrixPrevious[i] = rand() % nSize / (nSize * 1.0f);
		}
		// 最后一列0,0,0,1
		for(int i=0;i<nSize*JOINT_WIDTH;i++){
			if( (i/nSize+1)%4 == 0 )		pMatrixPrevious[i] = 0.0f;
			if( (i/nSize+1)%16 == 0 )	pMatrixPrevious[i] = 1.0f;
		}

		cudaMalloc( &pMatrixDevicePrevious, sizeof(float) * nSize * JOINT_WIDTH) ;
	}

	// 释放空间
	void unInitialize()
	{
		if (pMatrix) delete[] pMatrix;	
		if (pMatrixDevice) cudaFree(pMatrixDevice) ;
		
		if (pMatrixPrevious) delete[] pMatrixPrevious;
		if (pMatrixDevicePrevious) cudaFree(pMatrixDevicePrevious) ;
	}

	float		*pMatrix, *pMatrixDevice;
	float		*pMatrixPrevious, *pMatrixDevicePrevious; // 关节矩阵 上一帧
	int   nSize;// 关节的数目

};// 关节的集合

#endif // SEPERATE_STRUCT_FULLY

#else // SEPERATE_STRUCT

//关节矩阵---------------------------------------------------------
typedef Vector4  Matrix[MATRIX_SIZE_LINE];// 矩阵

struct Joints{

	// 获取关节矩阵
	void initialize( int size, float* pBufferMatrix ){
		
	}

	// 获取关节矩阵 模拟
	void initialize( int size ){
		nSize = size;
		pMatrix = new Matrix[nSize];
		for(int i=0;i<nSize;i++){
			for(int j=0;j<MATRIX_SIZE_LINE;j++){
				pMatrix[i][j].x = rand() % nSize / (nSize * 1.0f);
				pMatrix[i][j].y = rand() % nSize / (nSize * 1.0f);
				pMatrix[i][j].z = rand() % nSize / (nSize * 1.0f);
				if(i<3)	pMatrix[i][j].w = 0.0f;
				else		pMatrix[i][j].w = 1.0f;
			}
		}

		pMatrixPrevious = new Matrix[nSize];
		for(int i=0;i<nSize;i++){
			for(int j=0;j<MATRIX_SIZE_LINE;j++){
				pMatrixPrevious[i][j].x = rand() % nSize / (nSize * 1.0f);
				pMatrixPrevious[i][j].y = rand() % nSize / (nSize * 1.0f);
				pMatrixPrevious[i][j].z = rand() % nSize / (nSize * 1.0f);
				if(i<3)	pMatrixPrevious[i][j].w = 0.0f;
				else		pMatrixPrevious[i][j].w = 1.0f;
			}
		}

		cudaMalloc( &pMatrixDevice, sizeof(Matrix) * nSize ) ;
		cudaMalloc( &pMatrixDevicePrevious, sizeof(Matrix) * nSize ) ;
	}

	// 释放空间
	void unInitialize()
	{
		if (pMatrix) delete[] pMatrix;
		if (pMatrixPrevious) delete[] pMatrixPrevious;
		if (pMatrixDevice) cudaFree(pMatrixDevice) ;
		if (pMatrixDevicePrevious) cudaFree(pMatrixDevicePrevious) ;
	}

	Matrix*  pMatrix, *pMatrixDevice;
	Matrix*  pMatrixPrevious, *pMatrixDevicePrevious; // 关节矩阵 上一帧
	int   nSize;// 关节的数目

};// 关节的集合

#endif

#endif//JOINT_H__


