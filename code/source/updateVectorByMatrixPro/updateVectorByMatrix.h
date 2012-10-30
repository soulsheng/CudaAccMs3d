// updateVectorByMatrix.cpp : ���役�㺯��������任����
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

/* �������任
pVertex  : ����
pMatrix : ����
*/
void transformVec3ByMatrix4Host(float4* pVertexIn, float pMatrix[], float4* pVertexOut)
{
	float4 vertexIn = *pVertexIn;
	float4 vertexOut;
	vertexOut.x = vertexIn.x * pMatrix[0] + vertexIn.y * pMatrix[1] + vertexIn.z * pMatrix[2] + pMatrix[3] ; 
	vertexOut.y = vertexIn.x * pMatrix[1*4+0] + vertexIn.y * pMatrix[1*4+1] + vertexIn.z * pMatrix[1*4+2] + pMatrix[1*4+3]  ; 
	vertexOut.z = vertexIn.x * pMatrix[2*4+0] + vertexIn.y * pMatrix[2*4+1] + vertexIn.z * pMatrix[2*4+2] + pMatrix[2*4+3]  ;
	*pVertexOut = vertexOut;
}
template<typename T>
void transformVec3ByMatrix4Host(T* pVertexIn, T pMatrix[], T* pVertexOut)
{
	T vertexIn = *pVertexIn;
	T vertexOut;
	vertexOut.x = vertexIn.x * pMatrix[0].x + vertexIn.y * pMatrix[0].y + vertexIn.z * pMatrix[0].z + pMatrix[0].w ; 
	vertexOut.y = vertexIn.x * pMatrix[1].x + vertexIn.y * pMatrix[1].y + vertexIn.z * pMatrix[1].z + pMatrix[1].w  ; 
	vertexOut.z = vertexIn.x * pMatrix[2].x + vertexIn.y * pMatrix[2].y + vertexIn.z * pMatrix[2].z + pMatrix[2].w  ;
	*pVertexOut = vertexOut;
}

/* �任���� ����
pMatrix : ����
*/
void invertMatrix4Host(float mat[])
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

/* �任���� ���
pMatrix : ����
*/
void multMatrix4Host(float matIn1[], float matIn2[], float matOut[])
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


// ����������
template<typename T>
void indexByFloat44Host( T* pBuffer , T* pMat , int index )
{
	for(int j=0; j<MATRIX_SIZE_LINE; j++){
		pMat[j] = pBuffer[index * MATRIX_SIZE_LINE + j];
	}
}

// ������һ������
template<typename T>
void indexByFloat4Host( T* pBuffer , T* pMat , int index )
{
	for(int j=0; j<MATRIX_SIZE_LINE; j++){
		pMat[j] = pBuffer[index + MATRIX_SIZE_LINE * j];
	}
}

// ������һ����������
void indexByFloat1Host( float* pBuffer , float* pMat , int index )
{
	for(int i=0; i<JOINT_WIDTH; i++){
		pMat[ i ] = pBuffer[  i*JOINT_WIDTH + index ];
	}
}

/* �������任
pVertexIn  : ��̬���������������
size : �����������
pMatrix : �����������
pVertexOut : ��̬�������������
*/
#if !SEPERATE_STRUCT_FULLY
template<typename T>
void updateVectorByMatrixGold(T* pVertexIn, int size, Joints<T>* pJoints, T* pVertexOut, Index_Mode_Matrix mode){
#pragma omp parallel for
	for(int i=0;i<size;i++){
		
		T   matrix[MATRIX_SIZE_LINE];		

		// ��ȡ����������ʼ�Ķ�������
		T   vertexIn = pVertexIn[i];

		// ��ȡ�������������Ӧ�ľ���
		int      matrixIndex = int(vertexIn.w + 0.5);// float to int

		switch( mode )
		{
		case FLOAT_44:
			indexByFloat44Host( (T*)pJoints->pMatrix, matrix, matrixIndex );
			break;

		case FLOAT_4:
			indexByFloat4Host( (T*)pJoints->pMatrix, matrix, matrixIndex );
			break;

		case FLOAT_1:
			indexByFloat1Host( pJoints->pMatrix, (float*)matrix, matrixIndex );
			break;
		}

		// ִ�в�����������ִ�о���任���õ�������
		transformVec3ByMatrix4Host( &vertexIn, matrix, pVertexOut+i);
	}// for

}

template<typename T>
void updateVectorByMatrixGold(T* pVertexIn, int size, Joints<T>* pJoints, T* pVertexOut){
#pragma omp parallel for
	for(int i=0;i<size;i++){

		T   matrix[MATRIX_SIZE_LINE];
#if !USE_MEMORY_BUY_TIME
		float   matrixPrevious[JOINT_WIDTH];
		float   matrixCurrent[JOINT_WIDTH];
#endif


		// ��ȡ����������ʼ�Ķ�������
#if !USE_MEMORY_BUY_TIME
		T   vertexIn = pVertexOut[i];
#else
		T   vertexIn = pVertexIn[i];
#endif // USE_MEMORY_BUY_TIME

		// ��ȡ�������������Ӧ�ľ���
		int      matrixIndex = int(vertexIn.w + 0.5);// float to int
#if SEPERATE_STRUCT
#if USE_MEMORY_BUY_TIME
		for(int j=0; j<MATRIX_SIZE_LINE; j++){
			matrix[j] = pJoints->pMatrix[j*JOINT_SIZE+matrixIndex];
		}

#else
		for(int j=0; j<MATRIX_SIZE_LINE; j++){
			matrixCurrent[j*4+0] = pJoints->pMatrix[j*JOINT_SIZE+matrixIndex].x;
			matrixCurrent[j*4+1] = pJoints->pMatrix[j*JOINT_SIZE+matrixIndex].y;
			matrixCurrent[j*4+2] = pJoints->pMatrix[j*JOINT_SIZE+matrixIndex].z;
			matrixCurrent[j*4+3] = pJoints->pMatrix[j*JOINT_SIZE+matrixIndex].w;

			matrixPrevious[j*4+0] = pJoints->pMatrixPrevious[j*JOINT_SIZE+matrixIndex].x;
			matrixPrevious[j*4+1] = pJoints->pMatrixPrevious[j*JOINT_SIZE+matrixIndex].y;
			matrixPrevious[j*4+2] = pJoints->pMatrixPrevious[j*JOINT_SIZE+matrixIndex].z;
			matrixPrevious[j*4+3] = pJoints->pMatrixPrevious[j*JOINT_SIZE+matrixIndex].w;
		}
#endif // USE_MEMORY_BUY_TIME

#else// SEPERATE_STRUCT
#if USE_MEMORY_BUY_TIME
		for(int j=0; j<MATRIX_SIZE_LINE; j++){
			//matrix[j] = pJoints->pMatrix[matrixIndex * MATRIX_SIZE_LINE + j];
		}
#else
		for(int j=0; j<MATRIX_SIZE_LINE; j++){
			matrixCurrent[j*4+0] = pJoints->pMatrix[matrixIndex][j].x;
			matrixCurrent[j*4+1] = pJoints->pMatrix[matrixIndex][j].y;
			matrixCurrent[j*4+2] = pJoints->pMatrix[matrixIndex][j].z;
			matrixCurrent[j*4+3] = pJoints->pMatrix[matrixIndex][j].w;
		}
		for(int j=0; j<MATRIX_SIZE_LINE; j++){
			matrixPrevious[j*4+0] = pJoints->pMatrixPrevious[matrixIndex][j].x;
			matrixPrevious[j*4+1] = pJoints->pMatrixPrevious[matrixIndex][j].y;
			matrixPrevious[j*4+2] = pJoints->pMatrixPrevious[matrixIndex][j].z;
			matrixPrevious[j*4+3] = pJoints->pMatrixPrevious[matrixIndex][j].w;
		}
#endif // USE_MEMORY_BUY_TIME

#endif// SEPERATE_STRUCT


#if !USE_MEMORY_BUY_TIME
		invertMatrix4Host( matrixPrevious );
		multMatrix4Host( matrixCurrent, matrixPrevious, matrix );
#endif
		// ִ�в�����������ִ�о���任���õ�������
		transformVec3ByMatrix4Host( &vertexIn, matrix, pVertexOut+i);
	}

}

#else //SEPERATE_STRUCT_FULLY

void updateVectorByMatrixGoldFully(Vector4* pVertexIn, Vector4* pVertexOut, int size, float*pMatrix, float*pMatrixPrevious){
	for(int i=0;i<size;i++){

		float   matrix[JOINT_WIDTH];
#if !USE_MEMORY_BUY_TIME
		float   matrixPrevious[JOINT_WIDTH];
		float   matrixCurrent[JOINT_WIDTH];
#endif

		int      matrixIndex;

		// ��ȡ����������ʼ�Ķ�������
#if !USE_MEMORY_BUY_TIME
		Vector4   vertexIn = pVertexOut[i];
#else
		Vector4   vertexIn = pVertexIn[i];
#endif // USE_MEMORY_BUY_TIME

		// ��ȡ�������������Ӧ�ľ���
		matrixIndex = int(vertexIn.w + 0.5);// float to int
		
		for (int j=0;j<JOINT_WIDTH;j++)
		{
#if !USE_MEMORY_BUY_TIME
			matrixCurrent[j] = pMatrix[j*JOINT_SIZE+matrixIndex];
			matrixPrevious[j] = pMatrixPrevious[j*JOINT_SIZE+matrixIndex];
#else
			matrix[j] = pMatrix[j*JOINT_SIZE+matrixIndex];
#endif
		}

#if !USE_MEMORY_BUY_TIME
		invertMatrix4Host( matrixPrevious );
		multMatrix4Host( matrixCurrent, matrixPrevious, matrix );
#endif

		// ִ�в�����������ִ�о���任���õ�������
		transformVec3ByMatrix4Host( &vertexIn, matrix, pVertexOut+i);
	}

}

#endif //SEPERATE_STRUCT_FULLY


/* ��������Ƿ���ͬ
pVertex  : �������������
size : �������
pVertexBase : �ο���������
����ֵ�� 1��ʾ������ͬ��0��ʾ���겻ͬ
*/
template<typename T>
bool equalVector(T* pVertex, int size, T* pVertexBase)
{
	for(int i=0;i<size;i++)
	{
		T   vertex, vertexBase;
		vertex = pVertex[i];
		vertexBase = pVertexBase[i];
		if (fabs(vertex.x - vertexBase.x) / fabs(vertexBase.x) >1.7e-1 && fabs(vertex.x) * fabs(vertexBase.x) >10.0f || 
			fabs(vertex.y - vertexBase.y) / fabs(vertexBase.y) >1.7e-1 && fabs(vertex.y)  * fabs(vertexBase.y) >10.0f || 
			fabs(vertex.z - vertexBase.z) / fabs(vertexBase.z) >1.7e-1 && fabs(vertex.z)  * fabs(vertexBase.z) >10.0f ||
			fabs(vertexBase.x) >1.0e38 || fabs(vertexBase.y) >1.0e38 || fabs(vertexBase.z) >1.0e38 )
		{
			return false;
		}
	}

	return true;
}