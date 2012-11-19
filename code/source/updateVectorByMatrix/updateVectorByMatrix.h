// updateVectorByMatrix.cpp : ���役�㺯��������任����
//

#include "Vertex.h"
#include "Joint.h"
#include "Vector.h"
#include <omp.h>

/* �������任
pVertexIn  : ��̬���������������
size : �����������
pMatrix : �����������
pVertexOut : ��̬�������������
*/
void updateVectorByMatrix(Vertex* pVertexIn, int size, Matrix* pMatrix, Vertex* pVertexOut, bool use_openmp){

	if ( !use_openmp )
	{	
		omp_set_num_threads( 1 );
	}
#pragma omp parallel for
	for(int i=0;i<size;i++){
		float4   vertexIn, vertexOut;
		float4   matrix[3];
		int      matrixIndex;

		// ��ȡ����������ʼ�Ķ�������
		vertexIn = pVertexIn[i];

		// ��ȡ�������������Ӧ�ľ���
		matrixIndex = int(vertexIn.w + 0.5);// float to int
		matrix[0] = pMatrix[matrixIndex][0];
		matrix[1] = pMatrix[matrixIndex][1];
		matrix[2] = pMatrix[matrixIndex][2];

		// ִ�в�����������ִ�о���任���õ�������
		vertexOut.x = vertexIn.x * matrix[0].x + vertexIn.y * matrix[0].y + vertexIn.z * matrix[0].z  + matrix[0].w; 
		vertexOut.y = vertexIn.x * matrix[1].x + vertexIn.y * matrix[1].y + vertexIn.z * matrix[1].z  + matrix[1].w; 
		vertexOut.z = vertexIn.x * matrix[2].x + vertexIn.y * matrix[2].y + vertexIn.z * matrix[2].z  + matrix[2].w; 

		// д����������������
		pVertexOut[i] = vertexOut;
	}

}


void updateVectorByMatrixTest(Vertex* pVertexIn, int size, Matrix* pMatrix, Vertex* pVertexOut, bool use_openmp){

	if ( !use_openmp )
	{	
		omp_set_num_threads( 1 );
	}
#pragma omp parallel for
	for(int i=0;i<size;i++){
#if 0
		// ��ȡ�������������Ӧ�ľ���
		int matrixIndex = int(pVertexIn[i].w + 0.5);// float to int

		// ִ�в�����������ִ�о���任���õ�������
		pVertexOut[i].x = pVertexIn[i].x * pMatrix[matrixIndex][0].x + pVertexIn[i].y * pMatrix[matrixIndex][0].y + pVertexIn[i].z * pMatrix[matrixIndex][0].z  + pMatrix[matrixIndex][0].w; 
		pVertexOut[i].y = pVertexIn[i].x * pMatrix[matrixIndex][1].x + pVertexIn[i].y * pMatrix[matrixIndex][1].y + pVertexIn[i].z * pMatrix[matrixIndex][1].z  + pMatrix[matrixIndex][1].w; 
		pVertexOut[i].z = pVertexIn[i].x * pMatrix[matrixIndex][2].x + pVertexIn[i].y * pMatrix[matrixIndex][2].y + pVertexIn[i].z * pMatrix[matrixIndex][2].z  + pMatrix[matrixIndex][2].w; 
#else
		pVertexOut[i] = pVertexIn[i];
#endif
	}

}

template <typename F>
void testMaxInstruction(F* in, F* out, int size, bool use_openmp)
{
	if ( !use_openmp )
	{	
		omp_set_num_threads( 1 );
	}
#pragma omp parallel for
	for (int i=0;i<size;i++)
	{
		out[i] = in[i];
	}
}

void testMaxInstruction(float4* in, float4* out, int size, bool use_openmp)
{
	if ( !use_openmp )
	{	
		omp_set_num_threads( 1 );
	}
#pragma omp parallel for
	for (int i=0;i<size;i++)
	{
		out[i] = in[i];
	}
}

void updateVectorByMatrix(Vertex* in, Vertex* out, int size, bool use_openmp)
{
	if ( !use_openmp )
	{	
		omp_set_num_threads( 1 );
	}
#pragma omp parallel for
	for (int i=0;i<size;i++)
	{
		out[i] = in[i];
	}
}

