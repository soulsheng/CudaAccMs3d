// updateVectorByMatrix.cpp : 定义焦点函数，顶点变换矩阵
//

#include "Vertex.h"
#include "Joint.h"
#include "Vector.h"
#include <omp.h>
#include <xmmintrin.h>

bool verifyEqual(float *v, float* vRef, int size)
{
	for(int i=0;i<size*VERTEX_VECTOR_SIZE;i++)
	{
		//if ( (fabs(v[i]) - vRef[i]) / fabs(vRef[i]) >1.7e-1 && fabs(v[i]) * fabs(vRef[i]) >10.0f || fabs(v[i]) >1.0e38  )
		if ( (fabs(v[i]) - vRef[i]) / fabs(vRef[i]) >1e-3 )
		{
			return false;
		}
	}
	return true;
}
void MatrixVectorMul(float* vIn, float* vOut, float* mat)
{
		vOut[0] = vIn[0] * mat[0] + vIn[1] * mat[1] + vIn[2] * mat[2]  + mat[3];
		vOut[1] = vIn[0] * mat[4] + vIn[1] * mat[5] + vIn[2] * mat[6]  + mat[7];
		vOut[2] = vIn[0] * mat[8] + vIn[1] * mat[9] + vIn[2] * mat[10]  + mat[11];
}


/** Performing the transpose of a 4x4 matrix of single precision floating
    point values.
    Arguments r0, r1, r2, and r3 are __m128 values whose elements
    form the corresponding rows of a 4x4 matrix.
    The matrix transpose is returned in arguments r0, r1, r2, and
    r3 where r0 now holds column 0 of the original matrix, r1 now
    holds column 1 of the original matrix, etc.
*/
#define __MM_TRANSPOSE4x4_PS(r0, r1, r2, r3)                                            \
    {                                                                                   \
        __m128 tmp3, tmp2, tmp1, tmp0;                                                  \
                                                                                        \
                                                            /* r00 r01 r02 r03 */       \
                                                            /* r10 r11 r12 r13 */       \
                                                            /* r20 r21 r22 r23 */       \
                                                            /* r30 r31 r32 r33 */       \
                                                                                        \
        tmp0 = _mm_unpacklo_ps(r0, r1);                       /* r00 r10 r01 r11 */     \
        tmp2 = _mm_unpackhi_ps(r0, r1);                       /* r02 r12 r03 r13 */     \
        tmp1 = _mm_unpacklo_ps(r2, r3);                       /* r20 r30 r21 r31 */     \
        tmp3 = _mm_unpackhi_ps(r2, r3);                       /* r22 r32 r23 r33 */     \
                                                                                        \
        r0 = _mm_movelh_ps(tmp0, tmp1);                         /* r00 r10 r20 r30 */   \
        r1 = _mm_movehl_ps(tmp1, tmp0);                         /* r01 r11 r21 r31 */   \
        r2 = _mm_movelh_ps(tmp2, tmp3);                         /* r02 r12 r22 r32 */   \
        r3 = _mm_movehl_ps(tmp3, tmp2);                         /* r03 r13 r23 r33 */   \
    }

/// Accumulate four vector of single precision floating point values.
#define __MM_ACCUM4_PS(a, b, c, d)                                                  \
	_mm_add_ps(_mm_add_ps(a, b), _mm_add_ps(c, d))

/** Performing dot-product between four vector and three vector of single
    precision floating point values.
*/
#define __MM_DOT4x3_PS(r0, r1, r2, r3, v0, v1, v2)                                  \
    __MM_ACCUM4_PS(_mm_mul_ps(r0, v0), _mm_mul_ps(r1, v1), _mm_mul_ps(r2, v2), r3)


void updateVectorByMatrixSSE(float* pVertexIn, int* pIndex, int size, float* pMatrix, float* pVertexOut, bool use_openmp=false){

if (!use_openmp)
	omp_set_num_threads(1);

#pragma omp parallel for
	for(int i=0;i<size;i++){
		// 读取操作数：顶点对应的矩阵
	float *mat = pMatrix + pIndex[i]*MATRIX_SIZE_LINE*4;
	__m128 m0, m1, m2, m3;
	m0 = _mm_load_ps( mat );
    m1 = _mm_load_ps( mat+4 );
	m2 = _mm_load_ps( mat+8 );
	
	// Rearrange to column-major matrix with rows shuffled order to: Z 0 X Y
	m3 = _mm_setzero_ps();
	__MM_TRANSPOSE4x4_PS(m2, m3, m0, m1);

	// Load source position
	__m128 vI0, vI1, vI2;
	vI0 = _mm_load_ps1(pVertexIn +4*i );
	vI1 = _mm_load_ps1(pVertexIn  +4*i + 1);
	vI2 = _mm_load_ps1(pVertexIn  +4*i + 2);

	// Transform by collapsed matrix
	__m128 vO = __MM_DOT4x3_PS(m2, m3, m0, m1, vI0, vI1, vI2);   // z 0 x y
	
	// Store blended position, no aligned requirement
	_mm_storeh_pi((__m64*)(pVertexOut+4*i) , vO);
	_mm_store_ss(pVertexOut+4*i+2, vO);

	}
}

/* 坐标矩阵变换
pVertexIn  : 静态坐标数组参数输入
size : 坐标个数参数
pMatrix : 矩阵数组参数
pVertexOut : 动态坐标数组结果输出
*/
void updateVectorByMatrix(float* pVertexIn, int* pIndex, int size, float* pMatrix, float* pVertexOut, bool use_openmp=false){


#if 0//use_openmp
#pragma omp parallel for
#endif
	for(int i=0;i<size;i++){

		// 读取操作数：顶点对应的矩阵
		float *pMat = pMatrix + pIndex[i]*MATRIX_SIZE_LINE*4;
		
		// 执行操作：对坐标执行矩阵变换，得到新坐标
		MatrixVectorMul( pVertexIn+4*i, pVertexOut+4*i, pMat);

	}

}