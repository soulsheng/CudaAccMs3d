
#include "MatrixMulVector.h"
#include <omp.h>
#include "stdafx.h"

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


void CMatrixMulVector::ExecuteNativeSSE()
{
#if 0
	if (!USE_OPENMP)
		omp_set_num_threads(1);

#pragma omp parallel for
	for(int i=0;i<PROBLEM_SIZE;i++){
		// 读取操作数：顶点对应的矩阵
		float *mat = _joints.pMatrix + _vertexesStatic.pIndex[i]*MATRIX_SIZE_LINE*4;
		__m128 m0, m1, m2, m3;
		m0 = _mm_load_ps( mat );
		m1 = _mm_load_ps( mat+4 );
		m2 = _mm_load_ps( mat+8 );

		// Rearrange to column-major matrix with rows shuffled order to: Z 0 X Y
		m3 = _mm_setzero_ps();
		__MM_TRANSPOSE4x4_PS(m2, m3, m0, m1);

		// Load source position
		__m128 vI0, vI1, vI2;
		vI0 = _mm_load_ps1(_vertexesStatic.pVertex +4*i );
		vI1 = _mm_load_ps1(_vertexesStatic.pVertex  +4*i + 1);
		vI2 = _mm_load_ps1(_vertexesStatic.pVertex  +4*i + 2);

		// Transform by collapsed matrix
		__m128 vO = __MM_DOT4x3_PS(m2, m3, m0, m1, vI0, vI1, vI2);   // z 0 x y

		// Store blended position, no aligned requirement
		_mm_storeh_pi((__m64*)(_vertexesDynamicRef.pVertex+4*i) , vO);
		_mm_store_ss(_vertexesDynamicRef.pVertex+4*i+2, vO);

	}
#endif
}

void CMatrixMulVector::ExecuteNativeCPP()
{
#if 1//use_openmp
#pragma omp parallel for
#endif
	for(int i=0;i<_vertexesStatic.nSize;i++){

#if !VECTOR_FLOAT4
		// 读取操作数：顶点对应的矩阵
		float *pMat =  _joints.pMatrix + _vertexesStatic.pIndex[i]*MATRIX_SIZE_LINE*4;

		// 执行操作：对坐标执行矩阵变换，得到新坐标
		MatrixVectorMul( _vertexesStatic.pVertex+4*i, _vertexesDynamicRef.pVertex+4*i, pMat);
#else
		cl_float4 *pMat =  _joints.pMatrix + _vertexesStatic.pIndex[i]*MATRIX_SIZE_LINE;
		MatrixVectorMul( _vertexesStatic.pVertex+i, _vertexesDynamicRef.pVertex+i, pMat);
#endif
	}
}

bool CMatrixMulVector::verifyEqual( cl_float4 *v, cl_float4* vRef, int size )
{
	for(int i=0;i<size;i++)
	{
		for (int j=0;j<4;j++)
		{
			//if ( (fabs(v[i]) - vRef[i]) / fabs(vRef[i]) >1.7e-1 && fabs(v[i]) * fabs(vRef[i]) >10.0f || fabs(v[i]) >1.0e38  )
			if ( (fabs(v[i].s[j]) - vRef[i].s[j]) / fabs(vRef[i].s[j]) >1e-3 )
			{
				return false;
			}
		}

	}
	return true;
}

bool CMatrixMulVector::verifyEqual( float *v, float* vRef, int size )
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

bool CMatrixMulVector::verifyEqual()
{
	return verifyEqual( _vertexesDynamic.pVertex, _vertexesDynamicRef.pVertex,  _vertexesDynamic.nSize );
}


void CMatrixMulVector::unInitialize()
{
	_joints.unInitialize();
	_vertexesStatic.unInitialize();
	_vertexesDynamic.unInitialize();
}

void CMatrixMulVector::initialize(int sizeVertex, int sizeJoint)
{
	_joints.initialize( sizeJoint );
	_vertexesStatic.initialize( sizeVertex, sizeJoint );
	_vertexesDynamic.initialize( sizeVertex, sizeJoint );
	_vertexesDynamicRef.initialize( sizeVertex, sizeJoint );
}

CMatrixMulVector::~CMatrixMulVector()
{

}

CMatrixMulVector::CMatrixMulVector()
{

}

void CMatrixMulVector::MatrixVectorMul( float* vIn, float* vOut, float* mat )
{
	vOut[0] = vIn[0] * mat[0] + vIn[1] * mat[1] + vIn[2] * mat[2]  + mat[3];
	vOut[1] = vIn[0] * mat[4] + vIn[1] * mat[5] + vIn[2] * mat[6]  + mat[7];
	vOut[2] = vIn[0] * mat[8] + vIn[1] * mat[9] + vIn[2] * mat[10]  + mat[11];
}

void CMatrixMulVector::MatrixVectorMul( cl_float4* vIn, cl_float4* vOut, cl_float4* mat )
{
	vOut->s[0] = vIn->s[0] * mat[0].s[0] + vIn->s[1] * mat[0].s[1] + vIn->s[2] * mat[0].s[2]  + mat[0].s[3];
	vOut->s[1] = vIn->s[0] * mat[1].s[0] + vIn->s[1] * mat[1].s[1] + vIn->s[2] * mat[1].s[2]  + mat[1].s[3];
	vOut->s[2] = vIn->s[0] * mat[2].s[0] + vIn->s[1] * mat[2].s[1] + vIn->s[2] * mat[2].s[2]  + mat[2].s[3];
}
