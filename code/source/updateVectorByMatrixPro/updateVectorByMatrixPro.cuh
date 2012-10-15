// updateVectorByMatrixPro.cuh : 定义cuda kernel核函数
//

#include "Vertex.h"
#include "Joint.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"


#define USE_CONSTANT	0
__constant__ Matrix c_JointMatrix[ JOINT_SIZE ]; 

void constantMemoryUpdate( Matrix * pMatrix  , int nCountJoint)
{
    cudaMemcpyToSymbol((const char*)c_JointMatrix, pMatrix, nCountJoint * sizeof(Matrix) );
}

/* 坐标矩阵变换
pVertexIn  : 静态坐标数组参数输入
size : 坐标个数参数
pMatrix : 矩阵数组参数
pVertexOut : 动态坐标数组结果输出
*/
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Matrix* pMatrix, Vector4* pVertexOut)
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
#if USE_CONSTANT
		matrix[0] = c_JointMatrix[matrixIndex][0];
		matrix[1] = c_JointMatrix[matrixIndex][1];
		matrix[2] = c_JointMatrix[matrixIndex][2];
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
