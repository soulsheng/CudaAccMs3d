// updateVectorByMatrix.cpp : 定义焦点函数，顶点变换矩阵
//

#include "Vertex.h"
#include "Joint.h"


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

/* 坐标矩阵变换
pVertex  : 坐标
pMatrix : 矩阵
*/
void transformVec3ByMatrix4(float4* pVertexIn, float pMatrix[], float4* pVertexOut)
{
	float4 vertexIn = *pVertexIn;
	float4 vertexOut;
	vertexOut.x = vertexIn.x * pMatrix[0] + vertexIn.y * pMatrix[1] + vertexIn.z * pMatrix[2] + pMatrix[3] ; 
	vertexOut.y = vertexIn.x * pMatrix[1*4+0] + vertexIn.y * pMatrix[1*4+1] + vertexIn.z * pMatrix[1*4+2] + pMatrix[1*4+3]  ; 
	vertexOut.z = vertexIn.x * pMatrix[2*4+0] + vertexIn.y * pMatrix[2*4+1] + vertexIn.z * pMatrix[2*4+2] + pMatrix[2*4+3]  ;
	*pVertexOut = vertexOut;
}

/* 变换矩阵 求逆
pMatrix : 矩阵
*/
void invertMatrix4(float mat[])
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
void multMatrix4(float matIn1[], float matIn2[], float matOut[])
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
#if !SEPERATE_STRUCT_FULLY
void updateVectorByMatrixGold(Vector4* pVertexIn, int size, Joints* pJoints, Vector4* pVertexOut){
#pragma omp parallel for
	for(int i=0;i<size;i++){
		Vector4   vertexIn, vertexOut;
		Vector4   matrix[3];
#if !USE_MEMORY_BUY_TIME
		Vector4   matrixPrevious[3];
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
			// 执行操作：对坐标执行矩阵逆变换，得到初始坐标
			vertexOut.x = vertexIn.x * matrixPrevious[0].x + vertexIn.y * matrixPrevious[0].y + vertexIn.z * matrixPrevious[0].z + matrixPrevious[0].w ; 
			vertexOut.y = vertexIn.x * matrixPrevious[1].x + vertexIn.y * matrixPrevious[1].y + vertexIn.z * matrixPrevious[1].z + matrixPrevious[1].w  ; 
			vertexOut.z = vertexIn.x * matrixPrevious[2].x + vertexIn.y * matrixPrevious[2].y + vertexIn.z * matrixPrevious[2].z + matrixPrevious[2].w ; 

			vertexIn = vertexOut;
#endif // USE_MEMORY_BUY_TIME

		// 执行操作：对坐标执行矩阵变换，得到新坐标
		vertexOut.x = vertexIn.x * matrix[0].x + vertexIn.y * matrix[0].y + vertexIn.z * matrix[0].z + matrix[0].w ; 
		vertexOut.y = vertexIn.x * matrix[1].x + vertexIn.y * matrix[1].y + vertexIn.z * matrix[1].z + matrix[1].w  ; 
		vertexOut.z = vertexIn.x * matrix[2].x + vertexIn.y * matrix[2].y + vertexIn.z * matrix[2].z + matrix[2].w ; 

		// 写入操作结果：新坐标
		pVertexOut[i] = vertexOut;
	}

}

#else //SEPERATE_STRUCT_FULLY

void updateVectorByMatrixGoldFully(Vector4* pVertexIn, Vector4* pVertexOut, int size, float*pMatrix, float*pMatrixPrevious){
	for(int i=0;i<size;i++){

		float   matrix[JOINT_WIDTH];
		float   matrixInvert[JOINT_WIDTH];
		float   matrixIdentity[JOINT_WIDTH];
#if !USE_MEMORY_BUY_TIME
		float   matrixPrevious[JOINT_WIDTH];
#endif

		int      matrixIndex;

		// 读取操作数：初始的顶点坐标
#if !USE_MEMORY_BUY_TIME
		Vector4   vertexIn = pVertexOut[i];
#else
		Vector4   vertexIn = pVertexIn[i];
#endif // USE_MEMORY_BUY_TIME
		Vector4   vertexOut = vertexIn;

		// 读取操作数：顶点对应的矩阵
		matrixIndex = int(vertexIn.w + 0.5);// float to int
		
		for (int j=0;j<JOINT_WIDTH;j++)
		{
			matrix[j] = pMatrix[j*JOINT_SIZE+matrixIndex];
			matrixInvert[j] = pMatrix[j*JOINT_SIZE+matrixIndex];
#if !USE_MEMORY_BUY_TIME
			matrixPrevious[j] = pMatrixPrevious[j*JOINT_SIZE+matrixIndex];
#endif
		}

		invertMatrix4( matrixInvert );
		multMatrix4( matrix, matrixInvert, matrixIdentity );

#if !USE_MEMORY_BUY_TIME
		// 执行操作：对坐标执行矩阵逆变换，得到初始坐标
#if	USE_FUNCTION_TRANSFORM
		transformVec3ByMatrix4( &vertexIn, matrixPrevious, &vertexOut);
#else
		vertexOut.x = vertexIn.x * matrixPrevious[0] + vertexIn.y * matrixPrevious[1] + vertexIn.z * matrixPrevious[2] + matrixPrevious[3] ; 
		vertexOut.y = vertexIn.x * matrixPrevious[1*4+0] + vertexIn.y * matrixPrevious[1*4+1] + vertexIn.z * matrixPrevious[1*4+2] + matrixPrevious[1*4+3]  ; 
		vertexOut.z = vertexIn.x * matrixPrevious[2*4+0] + vertexIn.y * matrixPrevious[2*4+1] + vertexIn.z * matrixPrevious[2*4+2] + matrixPrevious[2*4+3]  ;

		vertexIn = vertexOut;
#endif// USE_FUNCTION_TRANSFORM
#endif// USE_MEMORY_BUY_TIME

		// 执行操作：对坐标执行矩阵变换，得到新坐标
#if	USE_FUNCTION_TRANSFORM
		transformVec3ByMatrix4( &vertexOut, matrix, pVertexOut+i);
#else
		vertexOut.x = vertexIn.x * matrix[0] + vertexIn.y * matrix[1] + vertexIn.z * matrix[2] + matrix[3] ; 
		vertexOut.y = vertexIn.x * matrix[1*4+0] + vertexIn.y * matrix[1*4+1] + vertexIn.z * matrix[1*4+2] + matrix[1*4+3]  ; 
		vertexOut.z = vertexIn.x * matrix[2*4+0] + vertexIn.y * matrix[2*4+1] + vertexIn.z * matrix[2*4+2] + matrix[2*4+3]  ;

		// 写入操作结果：新坐标
		pVertexOut[i] = vertexOut;
#endif
	}

}

#endif //SEPERATE_STRUCT_FULLY


/* 检测坐标是否相同
pVertex  : 待检测坐标数组
size : 坐标个数
pVertexBase : 参考坐标数组
返回值： 1表示坐标相同，0表示坐标不同
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
			fabs(vertex.z - vertexBase.z) / fabs(vertexBase.z) >1.0e-3 ||
			fabs(vertexBase.x) >1.0e38 || fabs(vertexBase.y) >1.0e38 || fabs(vertexBase.z) >1.0e38 )
		{
			return false;
		}
	}

	return true;
}