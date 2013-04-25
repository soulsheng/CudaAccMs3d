// updateVectorByMatrix.cpp : ���役�㺯��������任����
//

#include "Vertex.h"
#include "Joint.h"
#include "Vector.h"
#include <omp.h>

void MatrixVectorMul(float* vIn, float* vOut, float* mat)
{
	for (int i=0;i<3;i++)
	{
		vOut[i] = vIn[0] * mat[4*i + 0] + vIn[1] * mat[4*i + 1] + vIn[2] * mat[4*i + 2]  + mat[4*i + 3];
	}
}

/* �������任
pVertexIn  : ��̬���������������
size : �����������
pMatrix : �����������
pVertexOut : ��̬�������������
*/
void updateVectorByMatrix(float* pVertexIn, int* pIndex, int size, float* pMatrix, float* pVertexOut, bool use_openmp){

	if ( !use_openmp )
	{	
		omp_set_num_threads( 1 );
	}

#pragma omp parallel for
	for(int i=0;i<size;i++){

		// ��ȡ�������������Ӧ�ľ���
		float *pMat = pMatrix + pIndex[i]*MATRIX_SIZE_LINE*4;

		// ִ�в�����������ִ�о���任���õ�������
		MatrixVectorMul( pVertexIn+4*i, pVertexOut+4*i, pMat);

	}

}