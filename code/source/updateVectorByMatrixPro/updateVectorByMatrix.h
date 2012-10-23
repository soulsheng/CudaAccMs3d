// updateVectorByMatrix.cpp : 定义焦点函数，顶点变换矩阵
//

#include "Vertex.h"
#include "Joint.h"

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
		Vector4   vertexIn, vertexOut;
		float   matrix[JOINT_WIDTH];
#if !USE_MEMORY_BUY_TIME
		float   matrixPrevious[JOINT_WIDTH];
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
		
		for (int j=0;j<JOINT_WIDTH;j++)
		{
			matrix[j] = pMatrix[j*JOINT_SIZE+matrixIndex];
#if !USE_MEMORY_BUY_TIME
			matrixPrevious[j] = pMatrixPrevious[j*JOINT_SIZE+matrixIndex];
#endif
		}

#if !USE_MEMORY_BUY_TIME
		// 执行操作：对坐标执行矩阵逆变换，得到初始坐标
		vertexOut.x = vertexIn.x * matrixPrevious[0] + vertexIn.y * matrixPrevious[1] + vertexIn.z * matrixPrevious[2] + matrixPrevious[3] ; 
		vertexOut.y = vertexIn.x * matrixPrevious[1*4+0] + vertexIn.y * matrixPrevious[1*4+1] + vertexIn.z * matrixPrevious[1*4+2] + matrixPrevious[1*4+3]  ; 
		vertexOut.z = vertexIn.x * matrixPrevious[2*4+0] + vertexIn.y * matrixPrevious[2*4+1] + vertexIn.z * matrixPrevious[2*4+2] + matrixPrevious[2*4+3]  ;

		vertexIn = vertexOut;
#endif

		// 执行操作：对坐标执行矩阵变换，得到新坐标
		vertexOut.x = vertexIn.x * matrix[0] + vertexIn.y * matrix[1] + vertexIn.z * matrix[2] + matrix[3] ; 
		vertexOut.y = vertexIn.x * matrix[1*4+0] + vertexIn.y * matrix[1*4+1] + vertexIn.z * matrix[1*4+2] + matrix[1*4+3]  ; 
		vertexOut.z = vertexIn.x * matrix[2*4+0] + vertexIn.y * matrix[2*4+1] + vertexIn.z * matrix[2*4+2] + matrix[2*4+3]  ;

		// 写入操作结果：新坐标
		pVertexOut[i] = vertexOut;
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