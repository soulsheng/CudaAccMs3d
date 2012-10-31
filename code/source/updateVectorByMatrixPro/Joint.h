#ifndef JOINT_H__
#define JOINT_H__

#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"

//#define		ALIGNED_STRUCT		1// 对齐开关：0不对齐，1对齐
#define		USE_SHARED			0// 共享开关：0不共享，1共享
#define		SEPERATE_STRUCT	0// 结构体拆分开关：0不拆分，1拆分
#define		USE_MEMORY_BUY_TIME		1	// 以空间换时间， 1表示换，0表示不换（有bug）

#define		SEPERATE_STRUCT_FULLY		0 // 结构体彻底拆分开关：0不拆分，1拆分

#define    JOINT_SIZE    100
#define    MATRIX_SIZE_LINE    4//3
#define    JOINT_WIDTH    ((MATRIX_SIZE_LINE)*4)//12


struct Vector4 { float x,y,z,w; };

enum Matrix_Separate_Mode {
	NO_SEPARATE,		//	不拆分，相邻  1个float属于相邻矩阵的  1个float
	HALF_SEPARATE,		//	半拆分，相邻  4个float属于相邻矩阵的  4个float，矩阵一行
	COMPLETE_SEPARATE	//	全拆分，相邻16个float属于相邻矩阵的16个float，矩阵整体
};// 矩阵数组中相邻矩阵的存储方式


//关节矩阵---------------------------------------------------------

template<typename T>
struct Joints{

	// 初始化
	void initialize( int size , float** pBuffer, float** pBufferDevice ){
		
		// 设置矩阵个数，以及总字节数
		nSize = size;
		int nSizeFloat = JOINT_WIDTH * nSize;
		
		// 分配内存显存
		*pBuffer = new float[ nSizeFloat ];
		cudaMalloc( pBufferDevice, nSizeFloat * sizeof(float) ) ;
		
		switch( eSeparate )
		{
		case NO_SEPARATE:
			// 不拆分，按矩阵索引
			nSizePerElement = MATRIX_SIZE_LINE;
			indexByFloat44( pBuffer );
			break;

		case HALF_SEPARATE:
			// 半拆分，按矩阵一行索引
			nSizePerElement = MATRIX_SIZE_LINE;
			indexByFloat4( pBuffer );
			break;

		case COMPLETE_SEPARATE:
			// 全拆分，按矩阵一个浮点索引
			nSizePerElement = JOINT_WIDTH;
			indexByFloat1( pBuffer );
			break;
		}
		
	}


	// 按矩阵索引
	void indexByFloat44( float** pBuffer )
	{
		for(int i=0;i<nSize;i++){
			for(int j=0;j<nSizePerElement;j++){
				for(int k=0; k<4; k++ ){

					int index = 4*(i*nSizePerElement + j) + k;

					(* pBuffer)[ index ] = rand() % nSize / (nSize * 1.0f);

					if(k==3) {
						if(j<3)	(* pBuffer)[ index ] = 0.0f;
						else(* pBuffer)[ index ] = 1.0f;
					}//if k
				}//for k
			}//for j
		}//for i
	}

	// 按矩阵一行索引
	void indexByFloat4( float** pBuffer )
	{
		for(int i=0;i<nSizePerElement;i++){
			for(int j=0;j<nSize;j++){
				for(int k=0; k<4; k++ ){

					int index = 4*(i * nSize + j) + k;

					(* pBuffer)[index] = rand() % nSize / (nSize * 1.0f);

					if(k==3) {
						if(i<3)	(* pBuffer)[index] = 0.0f;
						else		(* pBuffer)[index] = 1.0f;
					}//if k
				}//for k
			}//for j
		}//for i
	}

	// 按矩阵一个浮点索引
	void indexByFloat1( float** pBuffer )
	{
		for(int i=0;i<nSizePerElement;i++){
			for(int j=0;j<nSize;j++){

					int index = i * nSize + j;

					(* pBuffer)[index] = rand() % nSize / (nSize * 1.0f);
					if( (i+1)%4 )		(* pBuffer)[index] = 0.0f;
					if( (i+1)%16 )		(* pBuffer)[index] = 1.0f;

			}//for j
		}//for i
	}

	// 获取关节矩阵 模拟
	void initialize( int size , Matrix_Separate_Mode mode )
	{
		eSeparate = mode;
		initialize( size, &pMatrix, &pMatrixDevice );
		initialize( size, &pMatrixPrevious, &pMatrixDevicePrevious );
	}

	// 释放空间
	void unInitialize( )
	{
		unInitialize( pMatrix, pMatrixDevice );
		unInitialize( pMatrixPrevious, pMatrixDevicePrevious );
	}

	// 释放空间
	void unInitialize( float* pBuffer, float* pBufferDevice )
	{
		if (pBuffer) delete[] pBuffer;
		if (pBufferDevice) cudaFree(pBufferDevice) ;
	}

	float*  pMatrix, *pMatrixDevice;
	float*  pMatrixPrevious, *pMatrixDevicePrevious; // 关节矩阵 上一帧
	int   nSize;// 关节的数目
	int   nSizePerElement;// 每个关节包含子数据结构的数目
	Matrix_Separate_Mode	eSeparate; // 索引矩阵数组的方式
};// 关节的集合


#endif//JOINT_H__


