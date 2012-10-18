// updateVectorByMatrixPro.cuh : 定义cuda kernel核函数
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

/* 坐标矩阵变换
pVertexIn  : 静态坐标数组参数输入
size : 坐标个数参数
pMatrix : 矩阵数组参数
pVertexOut : 动态坐标数组结果输出
*/
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

		// 读取操作数：初始的顶点坐标
		vertexIn = pVertexIn[i];

		// 读取操作数：顶点对应的矩阵
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

		// 执行操作：对坐标执行矩阵变换，得到新坐标
		vertexOut.x = vertexIn.x * matrix[0].x + vertexIn.y * matrix[0].y + vertexIn.z * matrix[0].z + matrix[0].w ; 
		vertexOut.y = vertexIn.x * matrix[1].x + vertexIn.y * matrix[1].y + vertexIn.z * matrix[1].z + matrix[1].w ; 
		vertexOut.z = vertexIn.x * matrix[2].x + vertexIn.y * matrix[2].y + vertexIn.z * matrix[2].z + matrix[2].w ; 

		// 写入操作结果：新坐标
		pVertexOut[i] = vertexOut;
	}
}

#if SEPERATE_STRUCT
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Vector4* pMatrix0, Vector4* pVertexOut, Vector4* pMatrix1, Vector4* pMatrix2, int sizeJoints)
#else
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Matrix* pMatrix, Vector4* pVertexOut, int sizeJoints)
#endif
{
	const int indexBase = blockIdx.x * blockDim.x + threadIdx.x;
	
	// 一次性读取矩阵，整个block块共享
	extern __shared__		float sharedMatrix[];
	if( threadIdx.x < sizeJoints )
	{
		Vector4   matrix[3];
#if SEPERATE_STRUCT
		matrix[0] = pMatrix0[threadIdx.x];
		matrix[1] = pMatrix1[threadIdx.x];
		matrix[2] = pMatrix2[threadIdx.x];
#else
		matrix[0] = pMatrix[threadIdx.x][0];
		matrix[1] = pMatrix[threadIdx.x][1];
		matrix[2] = pMatrix[threadIdx.x][2];
#endif

		int j=0;
		for(int i=0; i<3; i++)
		{
			sharedMatrix[j++] = matrix[i].x;
			sharedMatrix[j++] = matrix[i].y;
			sharedMatrix[j++] = matrix[i].z;
			sharedMatrix[j++] = matrix[i].w;
		}
	}
	//__syncthreads();

	for( int i=indexBase; i<size; i+=blockDim.x * gridDim.x ){
		Vector4   vertexIn, vertexOut;
		int      matrixIndex;

		// 读取操作数：初始的顶点坐标
		vertexIn = pVertexIn[i];

		// 读取操作数：顶点对应的矩阵
		matrixIndex = int(vertexIn.w + 0.5);// float to int

		// 执行操作：对坐标执行矩阵变换，得到新坐标
		vertexOut.x = vertexIn.x * sharedMatrix[0*4+0+matrixIndex] + vertexIn.y * sharedMatrix[0*4+1+matrixIndex] + vertexIn.z * sharedMatrix[0*4+2+matrixIndex] + sharedMatrix[0*4+3+matrixIndex] ; 
		vertexOut.y = vertexIn.x * sharedMatrix[1*4+0+matrixIndex] + vertexIn.y * sharedMatrix[1*4+1+matrixIndex] + vertexIn.z * sharedMatrix[1*4+2+matrixIndex] + sharedMatrix[1*4+3+matrixIndex] ; 
		vertexOut.z = vertexIn.x * sharedMatrix[2*4+0+matrixIndex] + vertexIn.y * sharedMatrix[2*4+1+matrixIndex] + vertexIn.z * sharedMatrix[2*4+2+matrixIndex] + sharedMatrix[2*4+3+matrixIndex] ; 

		// 写入操作结果：新坐标
		pVertexOut[i] = vertexOut;
	}
}