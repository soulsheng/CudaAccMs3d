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
void updateVectorByMatrixGold(Vector4* pVertexIn, int size, Matrix* pMatrix, Vector4* pVertexOut){
#pragma omp parallel for
	for(int i=0;i<size;i++){
		Vector4   vertexIn, vertexOut;
		Vector4   matrix[3];
		int      matrixIndex;

		// 读取操作数：初始的顶点坐标
		vertexIn = pVertexIn[i];

		// 读取操作数：顶点对应的矩阵
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

		// 执行操作：对坐标执行矩阵变换，得到新坐标
		vertexOut.x = vertexIn.x * matrix[0].x + vertexIn.y * matrix[0].y + vertexIn.z * matrix[0].z + matrix[0].w ; 
		vertexOut.y = vertexIn.x * matrix[1].x + vertexIn.y * matrix[1].y + vertexIn.z * matrix[1].z + matrix[1].w  ; 
		vertexOut.z = vertexIn.x * matrix[2].x + vertexIn.y * matrix[2].y + vertexIn.z * matrix[2].z + matrix[2].w ; 

		// 写入操作结果：新坐标
		pVertexOut[i] = vertexOut;
	}

}

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
		if (fabs(vertex.x - vertexBase.x) / vertexBase.x >1.0e-3 || 
			fabs(vertex.y - vertexBase.y) / vertexBase.y >1.0e-3 || 
			fabs(vertex.z - vertexBase.z) / vertexBase.z >1.0e-3 )
		{
			return false;
		}
	}

	return true;
}