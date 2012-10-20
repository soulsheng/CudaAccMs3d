// updateVectorByMatrixPro.cuh : ����cuda kernel�˺���
//

#include "Vertex.h"
#include "Joint.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"


void globalMemoryUpdate( Joints* pJoints )
{
#if SEPERATE_STRUCT
	for(int i=0;i<3;i++){
		cudaMemcpy( pJoints->pMatrixDevice[i], pJoints->pMatrix[i], sizeof(Vector4) * pJoints->nSize, cudaMemcpyHostToDevice );
	}
#else
	cudaMemcpy( pJoints->pMatrixDevice, pJoints->pMatrix, sizeof(Matrix) * pJoints->nSize, cudaMemcpyHostToDevice );
#endif
}

/* �������任
pVertexIn  : ��̬���������������
size : �����������
pMatrix : �����������
pVertexOut : ��̬�������������
*/
#if !USE_SHARED

#if SEPERATE_STRUCT
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Vector4* pMatrix0, Vector4* pVertexOut, Vector4* pMatrix1, Vector4* pMatrix2)
#else
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Matrix* pMatrix, Vector4* pVertexOut)
#endif
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
#if SEPERATE_STRUCT
		matrix[0] = pMatrix0[matrixIndex];
		matrix[1] = pMatrix1[matrixIndex];
		matrix[2] = pMatrix2[matrixIndex];
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

#else//USE_SHARED

#if SEPERATE_STRUCT
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Vector4* pMatrix0, Vector4* pVertexOut, Vector4* pMatrix1, Vector4* pMatrix2, int sizeJoints)
#else
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Matrix* pMatrix, Vector4* pVertexOut, int sizeJoints)
#endif
{
	const int indexBase = blockIdx.x * blockDim.x + threadIdx.x;
	
	// һ���Զ�ȡ��������block�鹲��
	__shared__		Vector4 matrix[3][JOINT_SIZE];
	if( threadIdx.x < sizeJoints )
	{
		for( int i=0;i<3;i++)
		{
			matrix[i][threadIdx.x] = pMatrix[threadIdx.x][i];
		}
	}
	__syncthreads();

	for( int i=indexBase; i<size; i+=blockDim.x * gridDim.x ){
		Vector4   vertexIn, vertexOut;
		int      matrixIndex;

		// ��ȡ����������ʼ�Ķ�������
		vertexIn = pVertexIn[i];

		// ��ȡ�������������Ӧ�ľ���
		matrixIndex = int(vertexIn.w + 0.5);// float to int

		// ִ�в�����������ִ�о���任���õ�������
		vertexOut.x = vertexIn.x * matrix[0][matrixIndex].x + vertexIn.y * matrix[0][matrixIndex].y + vertexIn.z * matrix[0][matrixIndex].z + matrix[0][matrixIndex].w ; 
		vertexOut.y = vertexIn.x * matrix[1][matrixIndex].x + vertexIn.y * matrix[1][matrixIndex].y + vertexIn.z * matrix[1][matrixIndex].z + matrix[1][matrixIndex].w  ; 
		vertexOut.z = vertexIn.x * matrix[2][matrixIndex].x + vertexIn.y * matrix[2][matrixIndex].y + vertexIn.z * matrix[2][matrixIndex].z + matrix[2][matrixIndex].w ; 

		// д����������������
		pVertexOut[i] = vertexOut;
	}
}

#endif//USE_SHARED