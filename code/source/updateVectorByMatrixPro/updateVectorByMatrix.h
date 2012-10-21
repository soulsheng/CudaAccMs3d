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
void updateVectorByMatrixGold(Vector4* pVertexIn, int size, Joints* pJoints, Vector4* pVertexOut){
#pragma omp parallel for
	for(int i=0;i<size;i++){
		Vector4   vertexIn, vertexOut;
		Vector4   matrix[3];
#if !USE_MEMORY_BUY_TIME
		Vector4   matrixPrevious[3];
#endif
		int      matrixIndex;

		// ��ȡ����������ʼ�Ķ�������
#if !USE_MEMORY_BUY_TIME
		vertexIn = pVertexOut[i];
#else
		vertexIn = pVertexIn[i];
#endif // USE_MEMORY_BUY_TIME

		// ��ȡ�������������Ӧ�ľ���
		matrixIndex = int(vertexIn.w + 0.5);// float to int
#if SEPERATE_STRUCT		
		matrix[0] = pJoints->pMatrix[0][matrixIndex];
		matrix[1] = pJoints->pMatrix[1][matrixIndex];
		matrix[2] = pJoints->pMatrix[2][matrixIndex];

	#if !USE_MEMORY_BUY_TIME
			matrixPrevious[0] = pJoints->pMatrixPrevious[0][matrixIndex];
			matrixPrevious[1] = pJoints->pMatrixPrevious[1][matrixIndex];
			matrixPrevious[2] = pJoints->pMatrixPrevious[2][matrixIndex];
	#endif // USE_MEMORY_BUY_TIME

#else
		matrix[0] = pJoints->pMatrix[matrixIndex][0];
		matrix[1] = pJoints->pMatrix[matrixIndex][1];
		matrix[2] = pJoints->pMatrix[matrixIndex][2];

	#if !USE_MEMORY_BUY_TIME
			matrixPrevious[0] = pJoints->pMatrixPrevious[matrixIndex][0];
			matrixPrevious[1] = pJoints->pMatrixPrevious[matrixIndex][1];
			matrixPrevious[2] = pJoints->pMatrixPrevious[matrixIndex][2];
	#endif // USE_MEMORY_BUY_TIME

#endif

#if !USE_MEMORY_BUY_TIME
			// ִ�в�����������ִ�о�����任���õ���ʼ����
			vertexOut.x = vertexIn.x * matrixPrevious[0].x + vertexIn.y * matrixPrevious[0].y + vertexIn.z * matrixPrevious[0].z + matrixPrevious[0].w ; 
			vertexOut.y = vertexIn.x * matrixPrevious[1].x + vertexIn.y * matrixPrevious[1].y + vertexIn.z * matrixPrevious[1].z + matrixPrevious[1].w  ; 
			vertexOut.z = vertexIn.x * matrixPrevious[2].x + vertexIn.y * matrixPrevious[2].y + vertexIn.z * matrixPrevious[2].z + matrixPrevious[2].w ; 

			vertexIn = vertexOut;
#endif // USE_MEMORY_BUY_TIME

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
		if (fabs(vertex.x - vertexBase.x) / fabs(vertexBase.x) >1.0e-3 || 
			fabs(vertex.y - vertexBase.y) / fabs(vertexBase.y) >1.0e-3 || 
			fabs(vertex.z - vertexBase.z) / fabs(vertexBase.z) >1.0e-3 )
		{
			return false;
		}
	}

	return true;
}