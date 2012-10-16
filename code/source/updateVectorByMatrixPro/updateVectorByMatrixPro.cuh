// updateVectorByMatrixPro.cuh : ����cuda kernel�˺���
//

#include "Vertex.h"
#include "Joint.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"


void globalMemoryUpdate( Matrix * pMatrix, Matrix * pMatrixDevice, int nCountJoint)
{
#if ALIGNED_STRUCT
	for(int i=0;i<3;i++){
		cudaMemcpy( (*pMatrixDevice)[i], (*pMatrix)[i], sizeof(Vector4) * nCountJoint, cudaMemcpyHostToDevice );
	}
#else
	cudaMemcpy( pMatrixDevice, pMatrix, sizeof(Matrix) * nCountJoint, cudaMemcpyHostToDevice );
#endif
}

/* �������任
pVertexIn  : ��̬���������������
size : �����������
pMatrix : �����������
pVertexOut : ��̬�������������
*/
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Matrix* pMatrix, Vector4* pVertexOut)
{
	const int indexBase = blockIdx.x * blockDim.x + threadIdx.x;
	for( int i=indexBase; i<size; i+=blockDim.x * gridDim.x ){
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
		vertexOut.y = vertexIn.x * matrix[1].x + vertexIn.y * matrix[1].y + vertexIn.z * matrix[1].z + matrix[1].w ; 
		vertexOut.z = vertexIn.x * matrix[2].x + vertexIn.y * matrix[2].y + vertexIn.z * matrix[2].z + matrix[2].w ; 

		// д����������������
		pVertexOut[i] = vertexOut;
	}
}
