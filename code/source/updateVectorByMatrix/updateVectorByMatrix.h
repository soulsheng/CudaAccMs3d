// updateVectorByMatrix.cpp : 定义焦点函数，顶点变换矩阵
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

/* 坐标矩阵变换
pVertexIn  : 静态坐标数组参数输入
size : 坐标个数参数
pMatrix : 矩阵数组参数
pVertexOut : 动态坐标数组结果输出
*/
void updateVectorByMatrix(float* pVertexIn, int* pIndex, int size, float* pMatrix, float* pVertexOut, bool use_openmp){

	if ( !use_openmp )
	{	
		omp_set_num_threads( 1 );
	}

#pragma omp parallel for
	for(int i=0;i<size;i++){

		// 读取操作数：顶点对应的矩阵
		float *pMat = pMatrix + pIndex[i]*MATRIX_SIZE_LINE*4;

		// 执行操作：对坐标执行矩阵变换，得到新坐标
		MatrixVectorMul( pVertexIn+4*i, pVertexOut+4*i, pMat);

	}

}