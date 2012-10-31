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

enum Matrix_Separate_Mode {
	NO_SEPARATE,		//	����֣�����  1��float�������ھ����  1��float
	HALF_SEPARATE,		//	���֣�����  4��float�������ھ����  4��float������һ��
	COMPLETE_SEPARATE	//	ȫ��֣�����16��float�������ھ����16��float����������
};// �������������ھ���Ĵ洢��ʽ


//�ؽھ���---------------------------------------------------------

template<typename T>
struct Joints{

	// ��ʼ��
	void initialize( int size , float** pBuffer, float** pBufferDevice ){
		
		// ���þ���������Լ����ֽ���
		nSize = size;
		int nSizeFloat = JOINT_WIDTH * nSize;
		
		// �����ڴ��Դ�
		*pBuffer = new float[ nSizeFloat ];
		cudaMalloc( pBufferDevice, nSizeFloat * sizeof(float) ) ;
		
		switch( eSeparate )
		{
		case NO_SEPARATE:
			// ����֣�����������
			nSizePerElement = MATRIX_SIZE_LINE;
			indexByFloat44( pBuffer );
			break;

		case HALF_SEPARATE:
			// ���֣�������һ������
			nSizePerElement = MATRIX_SIZE_LINE;
			indexByFloat4( pBuffer );
			break;

		case COMPLETE_SEPARATE:
			// ȫ��֣�������һ����������
			nSizePerElement = JOINT_WIDTH;
			indexByFloat1( pBuffer );
			break;
		}
		
	}


	// ����������
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

	// ������һ������
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

	// ������һ����������
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

	// ��ȡ�ؽھ��� ģ��
	void initialize( int size , Matrix_Separate_Mode mode )
	{
		eSeparate = mode;
		initialize( size, &pMatrix, &pMatrixDevice );
		initialize( size, &pMatrixPrevious, &pMatrixDevicePrevious );
	}

	// �ͷſռ�
	void unInitialize( )
	{
		unInitialize( pMatrix, pMatrixDevice );
		unInitialize( pMatrixPrevious, pMatrixDevicePrevious );
	}

	// �ͷſռ�
	void unInitialize( float* pBuffer, float* pBufferDevice )
	{
		if (pBuffer) delete[] pBuffer;
		if (pBufferDevice) cudaFree(pBufferDevice) ;
	}

	float*  pMatrix, *pMatrixDevice;
	float*  pMatrixPrevious, *pMatrixDevicePrevious; // �ؽھ��� ��һ֡
	int   nSize;// �ؽڵ���Ŀ
	int   nSizePerElement;// ÿ���ؽڰ��������ݽṹ����Ŀ
	Matrix_Separate_Mode	eSeparate; // ������������ķ�ʽ
};// �ؽڵļ���


#endif//JOINT_H__


