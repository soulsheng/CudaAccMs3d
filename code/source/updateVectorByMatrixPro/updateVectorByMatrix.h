// updateVectorByMatrix.cpp : ���役�㺯��������任����
//

#include "Vertex.h"
#include "Joint.h"

/* �������任
pVertexIn  : ��̬���������������
size : �����������
pMatrix : �����������
pVertexOut : ��̬�������������
*/
void updateVectorByMatrixGold(Vector4* pVertexIn, int size, Matrix* pMatrix, Vector4* pVertexOut){
#pragma omp parallel for
	for(int i=0;i<size;i++){
		Vector4   vertexIn, vertexOut;
		Vector4   matrix[3];
		int      matrixIndex;

		// ��ȡ����������ʼ�Ķ�������
		vertexIn = pVertexIn[i];

		// ��ȡ�������������Ӧ�ľ���
		matrixIndex = int(vertexIn.w + 0.5);// float to int
#if ALIGNED_STRUCT		
		matrix[0] = (*pMatrix)[0][matrixIndex];
		matrix[1] = (*pMatrix)[1][matrixIndex];
		matrix[2] = (*pMatrix)[2][matrixIndex];
#else
		matrix[0] = pMatrix[matrixIndex][0];
		matrix[1] = pMatrix[matrixIndex][1];
		matrix[2] = pMatrix[matrixIndex][2];
#endif

		// ִ�в�����������ִ�о���任���õ�������
		vertexOut.x = vertexIn.x * matrix[0].x + vertexIn.y * matrix[0].y + vertexIn.z * matrix[0].z + matrix[0].w ; 
		vertexOut.y = vertexIn.x * matrix[1].x + vertexIn.y * matrix[1].y + vertexIn.z * matrix[1].z + matrix[1].w  ; 
		vertexOut.z = vertexIn.x * matrix[2].x + vertexIn.y * matrix[2].y + vertexIn.z * matrix[2].z + matrix[2].w ; 

		// д����������������
		pVertexOut[i] = vertexOut;
	}

}

/* ��������Ƿ���ͬ
pVertex  : �������������
size : �������
pVertexBase : �ο���������
����ֵ�� 1��ʾ������ͬ��0��ʾ���겻ͬ
*/
bool equalVector(Vector4* pVertex, int size, Vector4* pVertexBase)
{
	for(int i=0;i<size;i++)
	{
		Vector4   vertex, vertexBase;
		vertex = pVertex[i];
		vertexBase = pVertexBase[i];
		if (fabs(vertex.x - vertexBase.x) / vertexBase.x >1.0e-3 || 
			fabs(vertex.y - vertexBase.y) / vertexBase.y >1.0e-3 || 
			fabs(vertex.z - vertexBase.z) / vertexBase.z >1.0e-3 )
		{
			return false;
		}
	}

	return true;
}