// updateVectorByMatrixPro.cuh : ����cuda kernel�˺���
//

#include "Vertex.h"
#include "Joint.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#define		USE_ELEMENT_CROSS	1	// ͬһ�̴߳���������Ԫ�أ� 1��ʾ��Ԫ�ؽ��棬0��ʾ�����漴˳��
#define		USE_ELEMENT_SINGLE	0	// ͬһ�̴߳���һ������Ԫ�أ� 1��ʾһ��Ԫ�أ�0��ʾ���Ԫ���Ҳ�����

__constant__		Vector4		const_pMatrix_v4[ JOINT_SIZE * MATRIX_SIZE_LINE ];
__constant__		Vector4		const_pMatrixPrevious_v4[ JOINT_SIZE * MATRIX_SIZE_LINE ];
__constant__		float4		const_pMatrix_f4[ JOINT_SIZE * MATRIX_SIZE_LINE ];
__constant__		float4		const_pMatrixPrevious_f4[ JOINT_SIZE * MATRIX_SIZE_LINE ];
__constant__		float			const_pMatrix_f1[ JOINT_SIZE * JOINT_WIDTH ];
__constant__		float			const_pMatrixPrevious_f1[ JOINT_SIZE * JOINT_WIDTH ];

template<typename F1>
void globalMemoryUpdate( Joints<F1>* pJoints, Matrix_Separate_Mode		modeSeparete, Matrix_Memory_Mode modeMemory, int bAligned )
{
	if( modeMemory == CONSTANT_MEMORY )
	{
		if( modeSeparete != COMPLETE_SEPARATE )
		{
			if( bAligned )
			{
				cudaMemcpyToSymbol( (char*)const_pMatrixPrevious_f4, (float*)pJoints->pMatrixPrevious, sizeof(float)*JOINT_WIDTH * pJoints->nSize );
				cudaMemcpyToSymbol( (char*)const_pMatrix_f4, (float*)pJoints->pMatrix, sizeof(float)*JOINT_WIDTH * pJoints->nSize );
			}
			else// !bAligned
			{
				cudaMemcpyToSymbol( (char*)const_pMatrixPrevious_v4, (float*)pJoints->pMatrixPrevious, sizeof(float)*JOINT_WIDTH * pJoints->nSize );
				cudaMemcpyToSymbol( (char*)const_pMatrix_v4, (float*)pJoints->pMatrix, sizeof(float)*JOINT_WIDTH * pJoints->nSize );
			}
		}
		else// COMPLETE_SEPARATE
		{
				cudaMemcpyToSymbol( (char*)const_pMatrixPrevious_f1, (float*)pJoints->pMatrixPrevious, sizeof(float)*JOINT_WIDTH * pJoints->nSize );
				cudaMemcpyToSymbol( (char*)const_pMatrix_f1, (float*)pJoints->pMatrix, sizeof(float)*JOINT_WIDTH * pJoints->nSize );
		}
	}
	else// !CONSTANT_MEMORY
	{
		cudaMemcpy( pJoints->pMatrixDevicePrevious, pJoints->pMatrixPrevious, sizeof(float)*JOINT_WIDTH * pJoints->nSize, cudaMemcpyHostToDevice );
		cudaMemcpy( pJoints->pMatrixDevice, pJoints->pMatrix, sizeof(float)*JOINT_WIDTH * pJoints->nSize, cudaMemcpyHostToDevice );
	}
}

/* �������任
pVertex  : ����
pMatrix : ����
*/
template<typename F4>
__device__ void transformVec3ByMatrix4(F4* pVertexIn, float1 pMatrix[], F4* pVertexOut)
{
	F4 vertexIn = *pVertexIn;
	F4 vertexOut;
	vertexOut.x = vertexIn.x * pMatrix[0] + vertexIn.y * pMatrix[1] + vertexIn.z * pMatrix[2] + pMatrix[3] ; 
	vertexOut.y = vertexIn.x * pMatrix[1*4+0] + vertexIn.y * pMatrix[1*4+1] + vertexIn.z * pMatrix[1*4+2] + pMatrix[1*4+3]  ; 
	vertexOut.z = vertexIn.x * pMatrix[2*4+0] + vertexIn.y * pMatrix[2*4+1] + vertexIn.z * pMatrix[2*4+2] + pMatrix[2*4+3]  ;
	*pVertexOut = vertexOut;
}
template<typename F4>
__device__ void transformVec3ByMatrix4(F4* pVertexIn, F4 pMatrix[], F4* pVertexOut)
{
	F4 vertexIn = *pVertexIn;
	F4 vertexOut;
	vertexOut.x = vertexIn.x * pMatrix[0].x + vertexIn.y * pMatrix[0].y + vertexIn.z * pMatrix[0].z + pMatrix[0].w ; 
	vertexOut.y = vertexIn.x * pMatrix[1].x + vertexIn.y * pMatrix[1].y + vertexIn.z * pMatrix[1].z + pMatrix[1].w  ; 
	vertexOut.z = vertexIn.x * pMatrix[2].x + vertexIn.y * pMatrix[2].y + vertexIn.z * pMatrix[2].z + pMatrix[2].w  ;
	*pVertexOut = vertexOut;
}

template<typename F4>
__device__ void transformVec3ByMatrix4_f4(F4* pVertexIn, float4 pMatrix[], F4* pVertexOut)
{
	F4 vertexIn = *pVertexIn;
	F4 vertexOut;
	vertexOut.x = vertexIn.x * pMatrix[0].x + vertexIn.y * pMatrix[0].y + vertexIn.z * pMatrix[0].z + pMatrix[0].w ; 
	vertexOut.y = vertexIn.x * pMatrix[1].x + vertexIn.y * pMatrix[1].y + vertexIn.z * pMatrix[1].z + pMatrix[1].w  ; 
	vertexOut.z = vertexIn.x * pMatrix[2].x + vertexIn.y * pMatrix[2].y + vertexIn.z * pMatrix[2].z + pMatrix[2].w  ;
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

/* �任���� ����
pMatrix : ����
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

/* �任���� ���
pMatrix : ����
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



	// ����������
template<typename F4>
__device__ void indexByFloat44( F4* pBuffer , F4* pMat , int index )
	{
		for(int j=0; j<MATRIX_SIZE_LINE; j++){
			pMat[j] = pBuffer[index * MATRIX_SIZE_LINE + j];
		}
	}

	// ������һ������
template<typename F4>
__device__ void indexByFloat4( F4* pBuffer , F4* pMat , int index )
	{
		for(int j=0; j<MATRIX_SIZE_LINE; j++){
			pMat[j] = pBuffer[index + JOINT_SIZE * j];
		}
	}

	// ������һ����������
template<typename F1>
__device__ void indexByFloat1( F1* pBuffer , F1* pMat , int index )
	{
		for(int j=0; j<JOINT_WIDTH; j++){
			pMat[j] = pBuffer[index + JOINT_SIZE * j];
		}
	}

/* �������任
pVertexIn  : ��̬���������������
size : �����������
pMatrix : �����������
pVertexOut : ��̬�������������
*/
#if !USE_SHARED

template<typename F4, typename F1>
__global__ void updateVectorByMatrix(F4* pVertexIn, int size, F1* pMatrix, F4* pVertexOut, F1* pMatrixPrevious, Matrix_Separate_Mode	modeSeparete)
{
	const int indexBase = ( gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;

		for( int i=indexBase; i<size; i+=blockDim.x * gridDim.x ){

		F4   matrix[MATRIX_SIZE_LINE];

		// ��ȡ����������ʼ�Ķ�������
		F4   vertexIn = pVertexIn[i];

		// ��ȡ�������������Ӧ�ľ���
		int      matrixIndex = int(vertexIn.w + 0.5);// float to int

		switch( modeSeparete )
		{
		case NO_SEPARATE:
			indexByFloat44( (F4*)pMatrix, matrix, matrixIndex );
			break;
		
		case HALF_SEPARATE:
			indexByFloat4( (F4*)pMatrix, matrix, matrixIndex );
			break;
		
		case COMPLETE_SEPARATE:
			indexByFloat1( pMatrix, (F1*)matrix, matrixIndex );
			break;
		}

		// ִ�в�����������ִ�о���任���õ�������
		transformVec3ByMatrix4( &vertexIn, matrix, pVertexOut+i);

	}//for
}


	// ����������
__device__ void indexMatrixConst( float4* pMat , int index, Matrix_Separate_Mode	modeSeparete, int bAligned )
	{
		// ��ɲ�֣�COMPLETE_SEPARATE
		if( modeSeparete==COMPLETE_SEPARATE )
		{
			float* tmpMat = (float*)pMat;
			for(int j=0; j<JOINT_WIDTH; j++)
				tmpMat[j] = const_pMatrix_f1[index + JOINT_SIZE * j];
		}
		else
		{// ����ֻ��߰��֣�NO_SEPARATE or HALF_SEPARATE

			for(int j=0; j<MATRIX_SIZE_LINE; j++)
			{
				int indexLast;

				if( modeSeparete==NO_SEPARATE )
					indexLast = index * MATRIX_SIZE_LINE + j;
				else if( modeSeparete==HALF_SEPARATE )
					indexLast = index + JOINT_SIZE * j;

				if( bAligned )
				{
					pMat[j] = const_pMatrix_f4[ indexLast ];
				}
				else
				{
					Vector4  tmpVector4 = const_pMatrix_v4[indexLast];
					pMat[j]	= make_float4( tmpVector4.x,  tmpVector4.y, tmpVector4.z, tmpVector4.w ) ;
				}
			}//for
		}//if else
	}

template<typename F4>
__global__ void updateVectorByMatrixConst(F4* pVertexIn, int size, F4* pVertexOut, Matrix_Separate_Mode	modeSeparete, int bAligned )
{
	const int indexBase = ( gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;

		for( int i=indexBase; i<size; i+=blockDim.x * gridDim.x ){

		float4   matrix[MATRIX_SIZE_LINE];

		// ��ȡ����������ʼ�Ķ�������
		F4   vertexIn = pVertexIn[i];

		// ��ȡ�������������Ӧ�ľ���
		int      matrixIndex = int(vertexIn.w + 0.5);// float to int

		indexMatrixConst( matrix, matrixIndex, modeSeparete, bAligned );

		// ִ�в�����������ִ�о���任���õ�������
		transformVec3ByMatrix4_f4( &vertexIn, matrix, pVertexOut+i);

	}//for
}

#else//USE_SHARED

#if SEPERATE_STRUCT
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Vector4* pMatrix, Vector4* pVertexOut,  Vector4* pMatrixPrevious)
#else
__global__ void updateVectorByMatrix(Vector4* pVertexIn, int size, Matrix* pMatrix, Vector4* pVertexOut, Matrix* pMatrixPrevious)
#endif
{
	const int indexBase = ( gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;
	
	// һ���Զ�ȡ��������block�鹲��
	__shared__		Vector4 matrix[MATRIX_SIZE_LINE][JOINT_SIZE];

	if( threadIdx.x < JOINT_SIZE )
	{
		float   matrixCurrent[JOINT_WIDTH];
		float   matrixPrevious[JOINT_WIDTH];
		float   matrixRegister[JOINT_WIDTH];
#if SEPERATE_STRUCT
#if !USE_MEMORY_BUY_TIME
		for (int j=0;j<MATRIX_SIZE_LINE;j++)
		{
			matrixCurrent[j*4+0] = pMatrix[j*JOINT_SIZE + threadIdx.x].x;
			matrixCurrent[j*4+1] = pMatrix[j*JOINT_SIZE + threadIdx.x].y;
			matrixCurrent[j*4+2] = pMatrix[j*JOINT_SIZE + threadIdx.x].z;
			matrixCurrent[j*4+3] = pMatrix[j*JOINT_SIZE + threadIdx.x].w;

			matrixPrevious[j*4+0] = pMatrixPrevious[j*JOINT_SIZE + threadIdx.x].x;
			matrixPrevious[j*4+1] = pMatrixPrevious[j*JOINT_SIZE + threadIdx.x].y;
			matrixPrevious[j*4+2] = pMatrixPrevious[j*JOINT_SIZE + threadIdx.x].z;
			matrixPrevious[j*4+3] = pMatrixPrevious[j*JOINT_SIZE + threadIdx.x].w;
		}
		
		invertMatrix4( matrixPrevious );
		multMatrix4( matrixCurrent, matrixPrevious, matrixRegister );
		
		for (int j=0;j<MATRIX_SIZE_LINE;j++)
		{
			matrix[j][threadIdx.x].x = matrixRegister[j*4+0];
			matrix[j][threadIdx.x].y = matrixRegister[j*4+1];
			matrix[j][threadIdx.x].z = matrixRegister[j*4+2];
			matrix[j][threadIdx.x].w = matrixRegister[j*4+3];
		}
#else
		for (int j=0;j<JOINT_WIDTH;j++)
		{
			matrix[j][threadIdx.x] = pMatrix[j*JOINT_SIZE + threadIdx.x];
		}
#endif
#else // !SEPERATE_STRUCT

#if USE_MEMORY_BUY_TIME
		for(int j=0; j<MATRIX_SIZE_LINE; j++){
			matrix[j*4+0] = pMatrix[matrixIndex][j].x;
			matrix[j*4+1] = pMatrix[matrixIndex][j].y;
			matrix[j*4+2] = pMatrix[matrixIndex][j].z;
			matrix[j*4+3] = pMatrix[matrixIndex][j].w;
		}
#else
		for(int j=0; j<MATRIX_SIZE_LINE; j++){
			matrixCurrent[j*4+0] = pMatrix[threadIdx.x][j].x;
			matrixCurrent[j*4+1] = pMatrix[threadIdx.x][j].y;
			matrixCurrent[j*4+2] = pMatrix[threadIdx.x][j].z;
			matrixCurrent[j*4+3] = pMatrix[threadIdx.x][j].w;
		}
		for(int j=0; j<MATRIX_SIZE_LINE; j++){
			matrixPrevious[j*4+0] = pMatrixPrevious[threadIdx.x][j].x;
			matrixPrevious[j*4+1] = pMatrixPrevious[threadIdx.x][j].y;
			matrixPrevious[j*4+2] = pMatrixPrevious[threadIdx.x][j].z;
			matrixPrevious[j*4+3] = pMatrixPrevious[threadIdx.x][j].w;
		}
			
		invertMatrix4( matrixPrevious );
		multMatrix4( matrixCurrent, matrixPrevious, matrixRegister );
		
		for (int j=0;j<MATRIX_SIZE_LINE;j++)
		{
			matrix[j][threadIdx.x].x = matrixRegister[j*4+0];
			matrix[j][threadIdx.x].y = matrixRegister[j*4+1];
			matrix[j][threadIdx.x].z = matrixRegister[j*4+2];
			matrix[j][threadIdx.x].w = matrixRegister[j*4+3];
		}
#endif // USE_MEMORY_BUY_TIME

#endif // SEPERATE_STRUCT
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

		// ��ȡ����������ʼ�Ķ�������
#if !USE_MEMORY_BUY_TIME
		Vector4   vertexIn = pVertexOut[i];
#else
		Vector4   vertexIn = pVertexIn[i];
#endif // USE_MEMORY_BUY_TIME

		// ��ȡ�������������Ӧ�ľ���
		int      matrixIndex = int(vertexIn.w + 0.5);// float to int
		
		float   matrixRegister[JOINT_WIDTH];
		for (int j=0;j<MATRIX_SIZE_LINE;j++)
		{
			matrixRegister[j*4 + 0] = matrix[j][matrixIndex].x;
			matrixRegister[j*4 + 1] = matrix[j][matrixIndex].y;
			matrixRegister[j*4 + 2] = matrix[j][matrixIndex].z;
			matrixRegister[j*4 + 3] = matrix[j][matrixIndex].w;
		}

		// ִ�в�����������ִ�о���任���õ�������
		transformVec3ByMatrix4( &vertexIn, matrixRegister, pVertexOut+i);

#if  !USE_ELEMENT_SINGLE
	}
#endif
}


__global__ void updateVectorByMatrixFully( Vector4* pVertexIn, Vector4* pVertexOut, int size, int sizeJoints, float* pMatrix, float* pMatrixPrevious)
{
	const int indexBase = ( gridDim.x * blockIdx.y + blockIdx.x ) * blockDim.x + threadIdx.x;

	// һ���Զ�ȡ��������block�鹲��
	__shared__		float matrix[JOINT_WIDTH][JOINT_SIZE];

	if( threadIdx.x < sizeJoints )
	{
#if !USE_MEMORY_BUY_TIME
		float   matrixCurrent[JOINT_WIDTH];
		float   matrixPrevious[JOINT_WIDTH];
		float   matrixRegister[JOINT_WIDTH];
		for (int j=0;j<JOINT_WIDTH;j++)
		{
			matrixCurrent[j] = pMatrix[j*JOINT_SIZE + threadIdx.x];
			matrixPrevious[j] = pMatrixPrevious[j*JOINT_SIZE + threadIdx.x];
		}
		invertMatrix4( matrixPrevious );
		multMatrix4( matrixCurrent, matrixPrevious, matrixRegister );
		for (int j=0;j<JOINT_WIDTH;j++)
		{
			matrix[j][threadIdx.x] = matrixRegister[j];
		}
#else
		for (int j=0;j<JOINT_WIDTH;j++)
		{
			matrix[j][threadIdx.x] = pMatrix[j*JOINT_SIZE + threadIdx.x];
		}
#endif
	}
	__syncthreads();

	for( int i=indexBase; i<size; i+=blockDim.x * gridDim.x ){

		// ��ȡ����������ʼ�Ķ�������
#if !USE_MEMORY_BUY_TIME
		Vector4   vertexIn = pVertexOut[i];
#else
		Vector4   vertexIn = pVertexIn[i];
#endif

		// ��ȡ�������������Ӧ�ľ���
		int      matrixIndex = int(vertexIn.w + 0.5);// float to int
		
		float   matrixRegister[JOINT_WIDTH];
		for (int j=0;j<JOINT_WIDTH;j++)
		{
			matrixRegister[j] = matrix[j][matrixIndex];
		}

		// ִ�в�����������ִ�о���任���õ�������
		transformVec3ByMatrix4( &vertexIn, matrixRegister, pVertexOut+i);

	}
}

#endif//USE_SHARED


template<typename F4>
__global__ void updateVectorByMatrix(F4* pVertexIn, int size, F4* pMatrix, F4* pVertexOut, F4* pMatrixPrevious)
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

		F4   matrix[MATRIX_SIZE_LINE];

#if !USE_MEMORY_BUY_TIME
		float   matrixPrevious[JOINT_WIDTH];
		float   matrixCurrent[JOINT_WIDTH];
#endif

		// ��ȡ����������ʼ�Ķ�������
#if !USE_MEMORY_BUY_TIME
		F4   vertexIn = pVertexOut[i];
#else
		F4   vertexIn = pVertexIn[i];
#endif // USE_MEMORY_BUY_TIME

		// ��ȡ�������������Ӧ�ľ���
		int      matrixIndex = int(vertexIn.w + 0.5);// float to int
#if SEPERATE_STRUCT
		for(int j=0; j<MATRIX_SIZE_LINE; j++){
			matrix[j] = pMatrix[matrixIndex + j*JOINT_SIZE];
		}

#if !USE_MEMORY_BUY_TIME
		matrixPrevious[0] = pMatrixPrevious0[matrixIndex];
		matrixPrevious[1] = pMatrixPrevious1[matrixIndex];
		matrixPrevious[2] = pMatrixPrevious2[matrixIndex];
#endif

#else
#if	USE_MEMORY_BUY_TIME
		for(int j=0; j<MATRIX_SIZE_LINE; j++){
			matrix[j] = pMatrix[matrixIndex*MATRIX_SIZE_LINE + j];
		}

#else
		for(int j=0; j<MATRIX_SIZE_LINE; j++){
			matrixCurrent[j*4+0] = pMatrix[matrixIndex][j].x;
			matrixCurrent[j*4+1] = pMatrix[matrixIndex][j].y;
			matrixCurrent[j*4+2] = pMatrix[matrixIndex][j].z;
			matrixCurrent[j*4+3] = pMatrix[matrixIndex][j].w;
		}
		for(int j=0; j<MATRIX_SIZE_LINE; j++){
			matrixPrevious[j*4+0] = pMatrixPrevious[matrixIndex][j].x;
			matrixPrevious[j*4+1] = pMatrixPrevious[matrixIndex][j].y;
			matrixPrevious[j*4+2] = pMatrixPrevious[matrixIndex][j].z;
			matrixPrevious[j*4+3] = pMatrixPrevious[matrixIndex][j].w;
		}
#endif
#endif

#if !USE_MEMORY_BUY_TIME
		invertMatrix4( matrixPrevious );
		multMatrix4( matrixCurrent, matrixPrevious, matrix );
#endif

		// ִ�в�����������ִ�о���任���õ�������
		transformVec3ByMatrix4( &vertexIn, matrix, pVertexOut+i);

#if  !USE_ELEMENT_SINGLE
	}
#endif
}