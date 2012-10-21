// updateVectorByMatrixPro.cuh : 定义cuda kernel核函数
//

#include "Vertex.h"
#include "Joint.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#define		USE_ELEMENT_CROSS	1	// 同一线程处理多个数据元素， 1表示多元素交替，0表示不交替即顺序
#define		USE_ELEMENT_SINGLE	0	// 同一线程处理一个数据元素， 1表示一个元素，0表示多个元素且不交替


void globalMemoryUpdate( Joints* pJoints )
{
#if SEPERATE_STRUCT
	for(int i=0;i<3;i++){
		cudaMemcpy( pJoints->pMatrixDevicePrevious[i], pJoints->pMatrixPrevious[i], sizeof(Vector4) * pJoints->nSize, cudaMemcpyHostToDevice );
		cudaMemcpy( pJoints->pMatrixDevice[i], pJoints->pMatrix[i], sizeof(Vector4) * pJoints->nSize, cudaMemcpyHostToDevice );
	}
#else
	cudaMemcpy( pJoints->pMatrixDevicePrevious, pJoints->pMatrixPrevious, sizeof(Matrix) * pJoints->nSize, cudaMemcpyHostToDevice );
	cudaMemcpy( pJoints->pMatrixDevice, pJoints->pMatrix, sizeof(Matrix) * pJoints->nSize, cudaMemcpyHostToDevice );
#endif
}

/* 坐标矩阵变换
pVertexIn  : 静态坐标数组参数输入
size : 坐标个数参数
pMatrix : 矩阵数组参数
pVertexOut : 动态坐标数组结果输出
*/
#if !USE_SHARED

#if SEPERATE_STRUCT
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Vector4* pMatrix0, Vector4* pVertexOut, Vector4* pMatrix1, Vector4* pMatrix2)
#else
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Matrix* pMatrix, Vector4* pVertexOut)
#endif
{
	const int indexBase = ( gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;

#if  !USE_ELEMENT_SINGLE
#if  !USE_ELEMENT_CROSS
	int nElementPerThread = (size+blockDim.x * gridDim.x-1)/(blockDim.x * gridDim.x);
	for( int j=0; j<nElementPerThread; j++ ){
		int i = indexBase * nElementPerThread + j;
		if( i >= size )
			break;
#else
		for( int i=indexBase; i<size; i+=blockDim.x * gridDim.x ){
#endif // USE_ELEMENT_CROSS

#else
		int i = indexBase;
		if( i >= size )
			return;
#endif // USE_ELEMENT_SINGLE

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
#if  !USE_ELEMENT_SINGLE
	}
#endif
}

#else//USE_SHARED

#if SEPERATE_STRUCT
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Vector4* pMatrix0, Vector4* pVertexOut, Vector4* pMatrix1, Vector4* pMatrix2, int sizeJoints,
															Vector4* pMatrixPrevious0, Vector4* pMatrixPrevious1, Vector4* pMatrixPrevious2)
#else
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Matrix* pMatrix, Vector4* pVertexOut, int sizeJoints, Matrix* pMatrixPrevious)
#endif
{
	const int indexBase = ( gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;
	
	// 一次性读取矩阵，整个block块共享
	__shared__		Vector4 matrix[3][JOINT_SIZE];
#if !USE_MEMORY_BUY_TIME
	__shared__		Vector4 matrixPrevious[3][JOINT_SIZE];
#endif
	if( threadIdx.x < sizeJoints )
	{
#if !SEPERATE_STRUCT
		for( int i=0;i<3;i++)
		{
			matrix[i][threadIdx.x] = pMatrix[threadIdx.x][i];
#if !USE_MEMORY_BUY_TIME
			matrixPrevious[i][threadIdx.x] = pMatrixPrevious[threadIdx.x][i];
#endif
		}
#else
		matrix[0][threadIdx.x] = pMatrix0[threadIdx.x];
		matrix[1][threadIdx.x] = pMatrix1[threadIdx.x];
		matrix[2][threadIdx.x] = pMatrix2[threadIdx.x];
		matrixPrevious[0][threadIdx.x] = pMatrixPrevious0[threadIdx.x];
		matrixPrevious[1][threadIdx.x] = pMatrixPrevious1[threadIdx.x];
		matrixPrevious[2][threadIdx.x] = pMatrixPrevious2[threadIdx.x];
#endif//!SEPERATE_STRUCT
	}
	__syncthreads();

#if  !USE_ELEMENT_SINGLE
#if  !USE_ELEMENT_CROSS
	int nElementPerThread = (size+blockDim.x * gridDim.x-1)/(blockDim.x * gridDim.x);
	for( int j=0; j<nElementPerThread; j++ ){
		int i = indexBase * nElementPerThread + j;
		if( i >= size )
			break;
#else
		for( int i=indexBase; i<size; i+=blockDim.x * gridDim.x ){
#endif // USE_ELEMENT_CROSS

#else
		int i = indexBase;
		if( i >= size )
			return;
#endif // USE_ELEMENT_SINGLE

		Vector4   vertexIn, vertexOut;
		int      matrixIndex;

		// 读取操作数：初始的顶点坐标
#if !USE_MEMORY_BUY_TIME
		vertexIn = pVertexOut[i];
#else
		vertexIn = pVertexIn[i];
#endif // USE_MEMORY_BUY_TIME

		// 读取操作数：顶点对应的矩阵
		matrixIndex = int(vertexIn.w + 0.5);// float to int

#if !USE_MEMORY_BUY_TIME
		// 执行操作：对坐标执行矩阵逆变换，得到初始坐标
		vertexOut.x = vertexIn.x * matrixPrevious[0][matrixIndex].x + vertexIn.y * matrixPrevious[0][matrixIndex].y + vertexIn.z * matrixPrevious[0][matrixIndex].z + matrixPrevious[0][matrixIndex].w ; 
		vertexOut.y = vertexIn.x * matrixPrevious[1][matrixIndex].x + vertexIn.y * matrixPrevious[1][matrixIndex].y + vertexIn.z * matrixPrevious[1][matrixIndex].z + matrixPrevious[1][matrixIndex].w  ; 
		vertexOut.z = vertexIn.x * matrixPrevious[2][matrixIndex].x + vertexIn.y * matrixPrevious[2][matrixIndex].y + vertexIn.z * matrixPrevious[2][matrixIndex].z + matrixPrevious[2][matrixIndex].w ; 
		
		vertexIn = vertexOut;
#endif // USE_MEMORY_BUY_TIME

		// 执行操作：对坐标执行矩阵变换，得到新坐标
		vertexOut.x = vertexIn.x * matrix[0][matrixIndex].x + vertexIn.y * matrix[0][matrixIndex].y + vertexIn.z * matrix[0][matrixIndex].z + matrix[0][matrixIndex].w ; 
		vertexOut.y = vertexIn.x * matrix[1][matrixIndex].x + vertexIn.y * matrix[1][matrixIndex].y + vertexIn.z * matrix[1][matrixIndex].z + matrix[1][matrixIndex].w  ; 
		vertexOut.z = vertexIn.x * matrix[2][matrixIndex].x + vertexIn.y * matrix[2][matrixIndex].y + vertexIn.z * matrix[2][matrixIndex].z + matrix[2][matrixIndex].w ; 

		// 写入操作结果：新坐标
		pVertexOut[i] = vertexOut;

#if  !USE_ELEMENT_SINGLE
	}
#endif
}

#endif//USE_SHARED