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
#if !SEPERATE_STRUCT_FULLY
	for(int i=0;i<3;i++){
		cudaMemcpy( pJoints->pMatrixDevicePrevious[i], pJoints->pMatrixPrevious[i], sizeof(Vector4) * pJoints->nSize, cudaMemcpyHostToDevice );
		cudaMemcpy( pJoints->pMatrixDevice[i], pJoints->pMatrix[i], sizeof(Vector4) * pJoints->nSize, cudaMemcpyHostToDevice );
	}
	#else
		cudaMemcpy( pJoints->pMatrixDevicePrevious, pJoints->pMatrixPrevious, sizeof(float)*JOINT_WIDTH * pJoints->nSize, cudaMemcpyHostToDevice );
		cudaMemcpy( pJoints->pMatrixDevice, pJoints->pMatrix, sizeof(float)*JOINT_WIDTH * pJoints->nSize, cudaMemcpyHostToDevice );
	#endif
#else
	cudaMemcpy( pJoints->pMatrixDevicePrevious, pJoints->pMatrixPrevious, sizeof(Matrix) * pJoints->nSize, cudaMemcpyHostToDevice );
	cudaMemcpy( pJoints->pMatrixDevice, pJoints->pMatrix, sizeof(Matrix) * pJoints->nSize, cudaMemcpyHostToDevice );
#endif
}

/* 坐标矩阵变换
pVertex  : 坐标
pMatrix : 矩阵
*/
__device__ void transformVec3ByMatrix4(float4* pVertexIn, float pMatrix[], float4* pVertexOut)
{
	float4 vertexIn = *pVertexIn;
	float4 vertexOut;
	vertexOut.x = vertexIn.x * pMatrix[0] + vertexIn.y * pMatrix[1] + vertexIn.z * pMatrix[2] + pMatrix[3] ; 
	vertexOut.y = vertexIn.x * pMatrix[1*4+0] + vertexIn.y * pMatrix[1*4+1] + vertexIn.z * pMatrix[1*4+2] + pMatrix[1*4+3]  ; 
	vertexOut.z = vertexIn.x * pMatrix[2*4+0] + vertexIn.y * pMatrix[2*4+1] + vertexIn.z * pMatrix[2*4+2] + pMatrix[2*4+3]  ;
	*pVertexOut = vertexOut;
}


#define SET_ROW( mat, row, v1, v2, v3, v4 )    \
	(mat)[(row)*4+0] = (v1); \
	(mat)[(row)*4+1] = (v2); \
	(mat)[(row)*4+2] = (v3); \
	(mat)[(row)*4+3] = (v4);

#define INNER_PRODUCT(matA,matB,r,c) \
	((matA)[r*4+0] * (matB)[0*4+c]) \
	+((matA)[r*4+1] * (matB)[1*4+c]) \
	+((matA)[r*4+2] * (matB)[2*4+c]) \
	+((matA)[r*4+3] * (matB)[3*4+c])

/* 变换矩阵 求逆
pMatrix : 矩阵
*/
__device__ void invertMatrix4(float mat[])
{
	float r00, r01, r02,
		r10, r11, r12,
		r20, r21, r22;
	// Copy rotation components directly into registers for speed
	r00 = mat[0*4+0]; r01 = mat[0*4+1]; r02 = mat[0*4+2];
	r10 = mat[1*4+0]; r11 = mat[1*4+1]; r12 = mat[1*4+2];
	r20 = mat[2*4+0]; r21 = mat[2*4+1]; r22 = mat[2*4+2];

	// Partially compute inverse of rot
	mat[0*4+0] = r11*r22 - r12*r21;
	mat[0*4+1] = r02*r21 - r01*r22;
	mat[0*4+2] = r01*r12 - r02*r11;

	// Compute determinant of rot from 3 elements just computed
	float one_over_det = 1.0/(r00*mat[0*4+0] + r10*mat[0*4+1] + r20*mat[0*4+2]);
	r00 *= one_over_det; r10 *= one_over_det; r20 *= one_over_det;  // Saves on later computations

	// Finish computing inverse of rot
	mat[0*4+0] *= one_over_det;
	mat[0*4+1] *= one_over_det;
	mat[0*4+2] *= one_over_det;
	mat[0*4+3] = 0.0;
	mat[1*4+0] = r12*r20 - r10*r22; // Have already been divided by det
	mat[1*4+1] = r00*r22 - r02*r20; // same
	mat[1*4+2] = r02*r10 - r00*r12; // same
	mat[1*4+3] = 0.0;
	mat[2*4+0] = r10*r21 - r11*r20; // Have already been divided by det
	mat[2*4+1] = r01*r20 - r00*r21; // same
	mat[2*4+2] = r00*r11 - r01*r10; // same
	mat[2*4+3] = 0.0;
	mat[3*4+3] = 1.0;

	float tx, ty, tz; 
	tx = mat[3*4+0]; ty = mat[3*4+1]; tz = mat[3*4+2];

	// Compute translation components of mat'
	mat[3*4+0] = -(tx*mat[0*4+0] + ty*mat[1*4+0] + tz*mat[2*4+0]);
	mat[3*4+1] = -(tx*mat[0*4+1] + ty*mat[1*4+1] + tz*mat[2*4+1]);
	mat[3*4+2] = -(tx*mat[0*4+2] + ty*mat[1*4+2] + tz*mat[2*4+2]);
}

/* 变换矩阵 相乘
pMatrix : 矩阵
*/
__device__ void multMatrix4(float matIn1[], float matIn2[], float matOut[])
{
	float t[4];
	for(int col=0; col<4; ++col) {
		t[0] = INNER_PRODUCT( matIn1, matIn2, 0, col );
		t[1] = INNER_PRODUCT( matIn1, matIn2, 1, col );
		t[2] = INNER_PRODUCT( matIn1, matIn2, 2, col );
		t[3] = INNER_PRODUCT( matIn1, matIn2, 3, col );
		matOut[0*4+col] = t[0];
		matOut[1*4+col] = t[1];
		matOut[2*4+col] = t[2];
		matOut[3*4+col] = t[3];
	}
}



/* 坐标矩阵变换
pVertexIn  : 静态坐标数组参数输入
size : 坐标个数参数
pMatrix : 矩阵数组参数
pVertexOut : 动态坐标数组结果输出
*/
#if !USE_SHARED

#if SEPERATE_STRUCT
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Vector4* pMatrix0, Vector4* pVertexOut, Vector4* pMatrix1, Vector4* pMatrix2, int sizeJoints,
															Vector4* pMatrixPrevious0, Vector4* pMatrixPrevious1, Vector4* pMatrixPrevious2)
#else
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Matrix* pMatrix, Vector4* pVertexOut, int sizeJoint, Matrix* pMatrixPrevious)
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
#if !USE_MEMORY_BUY_TIME
		Vector4		matrixPrevious[3];
#endif
		int      matrixIndex;

		// 读取操作数：初始的顶点坐标
#if !USE_MEMORY_BUY_TIME
		vertexIn = pVertexOut[i];
#else
		vertexIn = pVertexIn[i];
#endif // USE_MEMORY_BUY_TIME

		// 读取操作数：顶点对应的矩阵
		matrixIndex = int(vertexIn.w + 0.5);// float to int
#if SEPERATE_STRUCT
		matrix[0] = pMatrix0[matrixIndex];
		matrix[1] = pMatrix1[matrixIndex];
		matrix[2] = pMatrix2[matrixIndex];

#if !USE_MEMORY_BUY_TIME
		matrixPrevious[0] = pMatrixPrevious0[matrixIndex];
		matrixPrevious[1] = pMatrixPrevious1[matrixIndex];
		matrixPrevious[2] = pMatrixPrevious2[matrixIndex];
#endif

#else
		matrix[0] = pMatrix[matrixIndex][0];
		matrix[1] = pMatrix[matrixIndex][1];
		matrix[2] = pMatrix[matrixIndex][2];

#if !USE_MEMORY_BUY_TIME
		matrixPrevious[0] = pMatrixPrevious[matrixIndex][0];
		matrixPrevious[1] = pMatrixPrevious[matrixIndex][1];
		matrixPrevious[2] = pMatrixPrevious[matrixIndex][2];
#endif
#endif

#if !USE_MEMORY_BUY_TIME
		// 执行操作：对坐标执行矩阵逆变换，得到初始坐标
		vertexOut.x = vertexIn.x * matrixPrevious[0].x + vertexIn.y * matrixPrevious[0].y + vertexIn.z * matrixPrevious[0].z + matrixPrevious[0].w ; 
		vertexOut.y = vertexIn.x * matrixPrevious[1].x + vertexIn.y * matrixPrevious[1].y + vertexIn.z * matrixPrevious[1].z + matrixPrevious[1].w  ; 
		vertexOut.z = vertexIn.x * matrixPrevious[2].x + vertexIn.y * matrixPrevious[2].y + vertexIn.z * matrixPrevious[2].z + matrixPrevious[2].w ; 
		
		vertexIn = vertexOut;
#endif // USE_MEMORY_BUY_TIME

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


__global__ void updateVectorByMatrixFully( Vector4* pVertexIn, Vector4* pVertexOut, int size, int sizeJoints, float* pMatrix, float* pMatrixPrevious)
{
	const int indexBase = ( gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;

	for( int i=indexBase; i<size; i+=blockDim.x * gridDim.x ){

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
		
		float   matrix[JOINT_WIDTH];
		for (int j=0;j<JOINT_WIDTH;j++)
		{
			matrix[j]= pMatrix[j*JOINT_SIZE + matrixIndex];
		}

#if !USE_MEMORY_BUY_TIME
		float   matrixPrevious[JOINT_WIDTH];
		for (int j=0;j<JOINT_WIDTH;j++)
		{
			matrixPrevious[j]= pMatrixPrevious[j*JOINT_SIZE + matrixIndex];
		}
		// 执行操作：对坐标执行矩阵逆变换，得到初始坐标
		vertexOut.x = vertexIn.x * matrixPrevious[0] + vertexIn.y * matrixPrevious[1] + vertexIn.z * matrixPrevious[2] + matrixPrevious[3] ; 
		vertexOut.y = vertexIn.x * matrixPrevious[1*4+0] + vertexIn.y * matrixPrevious[1*4+1] + vertexIn.z * matrixPrevious[1*4+2] + matrixPrevious[1*4+3]  ; 
		vertexOut.z = vertexIn.x * matrixPrevious[2*4+0] + vertexIn.y * matrixPrevious[2*4+1] + vertexIn.z * matrixPrevious[2*4+2] + matrixPrevious[2*4+3]  ;

		vertexIn = vertexOut;
#endif
		// 执行操作：对坐标执行矩阵变换，得到新坐标
		vertexOut.x = vertexIn.x * matrix[0] + vertexIn.y * matrix[1] + vertexIn.z * matrix[2]+ matrix[3] ; 
		vertexOut.y = vertexIn.x * matrix[1*4+0] + vertexIn.y * matrix[1*4+1] + vertexIn.z * matrix[1*4+2] + matrix[1*4+3]  ; 
		vertexOut.z = vertexIn.x * matrix[2*4+0] + vertexIn.y * matrix[2*4+1]+ vertexIn.z * matrix[2*4+2] + matrix[2*4+3]  ;

		// 写入操作结果：新坐标
		pVertexOut[i] = vertexOut;
	}
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
#if !USE_MEMORY_BUY_TIME
		matrixPrevious[0][threadIdx.x] = pMatrixPrevious0[threadIdx.x];
		matrixPrevious[1][threadIdx.x] = pMatrixPrevious1[threadIdx.x];
		matrixPrevious[2][threadIdx.x] = pMatrixPrevious2[threadIdx.x];
#endif//!USE_MEMORY_BUY_TIME
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


__global__ void updateVectorByMatrixFully( Vector4* pVertexIn, Vector4* pVertexOut, int size, int sizeJoints, float* pMatrix, float* pMatrixPrevious)
{
	const int indexBase = ( gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;

	// 一次性读取矩阵，整个block块共享
	__shared__		float matrixCurrent[JOINT_WIDTH][JOINT_SIZE];
#if !USE_MEMORY_BUY_TIME
	__shared__		float matrixPrevious[JOINT_WIDTH][JOINT_SIZE];
#endif

	if( threadIdx.x < sizeJoints )
	{
		for (int j=0;j<JOINT_WIDTH;j++)
		{
			matrixCurrent[j][threadIdx.x] = pMatrix[j*JOINT_SIZE + threadIdx.x];
#if !USE_MEMORY_BUY_TIME
			matrixPrevious[j][threadIdx.x] = pMatrixPrevious[j*JOINT_SIZE + threadIdx.x];
#endif
		}
	}
	__syncthreads();

	for( int i=indexBase; i<size; i+=blockDim.x * gridDim.x ){

		// 读取操作数：初始的顶点坐标
#if !USE_MEMORY_BUY_TIME
		Vector4   vertexIn = pVertexOut[i];
#else
		Vector4   vertexIn = pVertexIn[i];
		Vector4		vertexOut = vertexIn;
#endif

		// 读取操作数：顶点对应的矩阵
		int      matrixIndex = int(vertexIn.w + 0.5);// float to int
		
		float   matrixCurrentRegister[JOINT_WIDTH];
		float   matrixPreviousRegister[JOINT_WIDTH];
		float   matrixRegister[JOINT_WIDTH];
		for (int j=0;j<JOINT_WIDTH;j++)
		{
			matrixCurrentRegister[j] = matrixCurrent[j][matrixIndex];
#if !USE_MEMORY_BUY_TIME
			matrixPreviousRegister[j] = matrixPrevious[j][matrixIndex];
#endif
		}
		
		invertMatrix4( matrixPreviousRegister );
		multMatrix4( matrixCurrentRegister, matrixPreviousRegister, matrixRegister );

		// 执行操作：对坐标执行矩阵变换，得到新坐标
#if USE_FUNCTION_TRANSFORM
		transformVec3ByMatrix4( &vertexIn, matrixRegister, pVertexOut+i);
#else
		vertexOut.x = vertexIn.x * matrix[0][matrixIndex] + vertexIn.y * matrix[1][matrixIndex] + vertexIn.z * matrix[2][matrixIndex] + matrix[3][matrixIndex] ; 
		vertexOut.y = vertexIn.x * matrix[1*4+0][matrixIndex] + vertexIn.y * matrix[1*4+1][matrixIndex] + vertexIn.z * matrix[1*4+2][matrixIndex] + matrix[1*4+3][matrixIndex]  ; 
		vertexOut.z = vertexIn.x * matrix[2*4+0][matrixIndex] + vertexIn.y * matrix[2*4+1][matrixIndex] + vertexIn.z * matrix[2*4+2][matrixIndex] + matrix[2*4+3][matrixIndex]  ;

		// 写入操作结果：新坐标
		pVertexOut[i] = vertexOut;
#endif
	}
}

#endif//USE_SHARED
