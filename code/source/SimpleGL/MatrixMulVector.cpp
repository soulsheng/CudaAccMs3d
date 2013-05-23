
#include "MatrixMulVector.h"
#include <omp.h>
//#include "stdafx.h"
#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <CL/cl_gl.h>

#define  LocalWorkX		8
#define  LocalWorkY		8

#define  VBO_MAP		1
#define  USE_OPENMP		0

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

/// Same as _mm_load_ps, but can help VC generate more optimised code.
#define __MM_LOAD_PS(p)                                                             \
	(*(__m128*)(p))

/// Same as _mm_store_ps, but can help VC generate more optimised code.
#define __MM_STORE_PS(p, v)                                                         \
	(*(__m128*)(p) = (v))


/// Calculate multiply of two vector and plus another vector
#define __MM_MADD_PS(a, b, c)                                                       \
	_mm_add_ps(_mm_mul_ps(a, b), c)

/// Linear interpolation
#define __MM_LERP_PS(t, a, b)                                                       \
	__MM_MADD_PS(_mm_sub_ps(b, a), t, a)
//---------------------------------------------------------------------
// Some useful macro for collapse matrices.
//---------------------------------------------------------------------
#if STRUCT_OCL
#define __LOAD_MATRIX(row0, row1, row2, pMatrix)                        \
	{                                                                   \
	row0 = __MM_LOAD_PS(pMatrix[0]);                             \
	row1 = __MM_LOAD_PS(pMatrix[1]);                             \
	row2 = __MM_LOAD_PS(pMatrix[2]);                             \
	}

#define __LERP_MATRIX(row0, row1, row2, weight, pMatrix)                \
	{                                                                   \
	row0 = __MM_LERP_PS(weight, row0, __MM_LOAD_PS(pMatrix[0])); \
	row1 = __MM_LERP_PS(weight, row1, __MM_LOAD_PS(pMatrix[1])); \
	row2 = __MM_LERP_PS(weight, row2, __MM_LOAD_PS(pMatrix[2])); \
	}

#define __LOAD_WEIGHTED_MATRIX(row0, row1, row2, weight, pMatrix)       \
	{                                                                   \
	row0 = _mm_mul_ps(__MM_LOAD_PS(pMatrix[0]), weight);         \
	row1 = _mm_mul_ps(__MM_LOAD_PS(pMatrix[1]), weight);         \
	row2 = _mm_mul_ps(__MM_LOAD_PS(pMatrix[2]), weight);         \
	}

#define __ACCUM_WEIGHTED_MATRIX(row0, row1, row2, weight, pMatrix)      \
	{                                                                   \
	row0 = __MM_MADD_PS(__MM_LOAD_PS(pMatrix[0]), weight, row0); \
	row1 = __MM_MADD_PS(__MM_LOAD_PS(pMatrix[1]), weight, row1); \
	row2 = __MM_MADD_PS(__MM_LOAD_PS(pMatrix[2]), weight, row2); \
	}
#else

#define __LOAD_MATRIX(row0, row1, row2, pMatrix)                        \
	{                                                                   \
	row0 = __MM_LOAD_PS(pMatrix);                             \
	row1 = __MM_LOAD_PS(pMatrix+4);                             \
	row2 = __MM_LOAD_PS(pMatrix+8);                             \
	}

#define __LERP_MATRIX(row0, row1, row2, weight, pMatrix)                \
	{                                                                   \
	row0 = __MM_LERP_PS(weight, row0, __MM_LOAD_PS(pMatrix)); \
	row1 = __MM_LERP_PS(weight, row1, __MM_LOAD_PS(pMatrix+4)); \
	row2 = __MM_LERP_PS(weight, row2, __MM_LOAD_PS(pMatrix+8)); \
	}

#define __LOAD_WEIGHTED_MATRIX(row0, row1, row2, weight, pMatrix)       \
	{                                                                   \
	row0 = _mm_mul_ps(__MM_LOAD_PS(pMatrix), weight);         \
	row1 = _mm_mul_ps(__MM_LOAD_PS(pMatrix+4), weight);         \
	row2 = _mm_mul_ps(__MM_LOAD_PS(pMatrix+8), weight);         \
	}

#define __ACCUM_WEIGHTED_MATRIX(row0, row1, row2, weight, pMatrix)      \
	{                                                                   \
	row0 = __MM_MADD_PS(__MM_LOAD_PS(pMatrix), weight, row0); \
	row1 = __MM_MADD_PS(__MM_LOAD_PS(pMatrix+4), weight, row1); \
	row2 = __MM_MADD_PS(__MM_LOAD_PS(pMatrix+8), weight, row2); \
	}

#endif
/** Fill vector of single precision floating point with selected value.
    Argument 'fp' is a digit[0123] that represents the fp of argument 'v'.
*/
#define __MM_SELECT(v, fp)                                                          \
    _mm_shuffle_ps((v), (v), _MM_SHUFFLE((fp),(fp),(fp),(fp)))

/** Collapse one-weighted matrix.
    Eliminated multiply by weight since the weight should be equal to one always
*/
#define __COLLAPSE_MATRIX_W1(row0, row1, row2, ppMatrices, pIndices, pWeights)  \
    {                                                                           \
        pMatrix0 = blendMatrices +pIndices[0]*MATRIX_SIZE_LINE;                                  \
        __LOAD_MATRIX(row0, row1, row2, pMatrix0);                              \
    }

/** Collapse two-weighted matrix.
    Based on the fact that accumulated weights are equal to one, by use lerp,
    replaced two multiplies and one additive with one multiplie and two additives.
*/
#define __COLLAPSE_MATRIX_W2(row0, row1, row2, ppMatrices, pIndices, pWeights)  \
    {                                                                           \
        weight = _mm_load_ps1(pWeights + 1);                                    \
        pMatrix0 = ppMatrices +pIndices[0]*MATRIX_SIZE_LINE;                                     \
        __LOAD_MATRIX(row0, row1, row2, pMatrix0);                              \
        pMatrix1 = ppMatrices +pIndices[1]*MATRIX_SIZE_LINE;                                     \
        __LERP_MATRIX(row0, row1, row2, weight, pMatrix1);                      \
    }

/** Collapse three-weighted matrix.
*/
#define __COLLAPSE_MATRIX_W3(row0, row1, row2, ppMatrices, pIndices, pWeights)  \
    {                                                                           \
        weight = _mm_load_ps1(pWeights + 0);                                    \
        pMatrix0 = ppMatrices + pIndices[0]*MATRIX_SIZE_LINE;                                     \
        __LOAD_WEIGHTED_MATRIX(row0, row1, row2, weight, pMatrix0);             \
        weight = _mm_load_ps1(pWeights + 1);                                    \
        pMatrix1 = ppMatrices + pIndices[1]*MATRIX_SIZE_LINE;                                     \
        __ACCUM_WEIGHTED_MATRIX(row0, row1, row2, weight, pMatrix1);            \
        weight = _mm_load_ps1(pWeights + 2);                                    \
        pMatrix2 = ppMatrices + pIndices[2]*MATRIX_SIZE_LINE;                                     \
        __ACCUM_WEIGHTED_MATRIX(row0, row1, row2, weight, pMatrix2);            \
    }

/** Collapse four-weighted matrix.
*/
#define __COLLAPSE_MATRIX_W4(row0, row1, row2, ppMatrices, pIndices, pWeights)  \
    {                                                                           \
        /* Load four blend weights at one time, they will be shuffled later */  \
        weights = _mm_loadu_ps(pWeights);                                       \
                                                                                \
        pMatrix0 = ppMatrices + pIndices[0]*MATRIX_SIZE_LINE;                                     \
        weight = __MM_SELECT(weights, 0);                                       \
        __LOAD_WEIGHTED_MATRIX(row0, row1, row2, weight, pMatrix0);             \
        pMatrix1 = ppMatrices + pIndices[1]*MATRIX_SIZE_LINE;                                     \
        weight = __MM_SELECT(weights, 1);                                       \
        __ACCUM_WEIGHTED_MATRIX(row0, row1, row2, weight, pMatrix1);            \
        pMatrix2 = ppMatrices + pIndices[2]*MATRIX_SIZE_LINE;                                     \
        weight = __MM_SELECT(weights, 2);                                       \
        __ACCUM_WEIGHTED_MATRIX(row0, row1, row2, weight, pMatrix2);            \
        pMatrix3 = ppMatrices + pIndices[3]*MATRIX_SIZE_LINE;                                     \
        weight = __MM_SELECT(weights, 3);                                       \
        __ACCUM_WEIGHTED_MATRIX(row0, row1, row2, weight, pMatrix3);            \
    }



    //---------------------------------------------------------------------
    // Collapse a matrix at one time. The collapsed matrix are weighted by
    // blend-weights, and then can use to transform corresponding vertex directly.
    //
    // I'd like use inline function instead of macro here, but I also want to
    // ensure compiler integrate this code into its callers (release build at
    // least), doesn't matter about specific compile options. Inline function
    // work fine for VC, but looks like gcc (3.4.4 here) generate function-call
    // when implemented as inline function, even if compile with "-O3" option.
    //
#define _collapseOneMatrix(                                                     \
        m00, m01, m02,                                                          \
        pBlendWeight, pBlendIndex,                                              \
        blendMatrices,                                                          \
        blendWeightStride, blendIndexStride,                                    \
        numWeightsPerVertex)                                                    \
    {                                                                           \
        /* Important Note: If reuse pMatrixXXX frequently, M$ VC7.1 will */     \
        /* generate wrong code here!!!                                   */     \
        float * pMatrix0, *pMatrix1, *pMatrix2, *pMatrix3;               \
        __m128 weight, weights;                                                 \
                                                                                \
        switch (numWeightsPerVertex)                                            \
        {                                                                       \
        default:    /* Just in case and make compiler happy */                  \
        case 1:                                                                 \
            __COLLAPSE_MATRIX_W1(m00, m01, m02, blendMatrices, pBlendIndex, pBlendWeight);         \
            break;                                                              \
                                                                                \
        case 2:                                                                 \
            __COLLAPSE_MATRIX_W2(m00, m01, m02, blendMatrices,  pBlendIndex, pBlendWeight);         \
            break;                                                              \
                                                                                \
        case 3:                                                                 \
            __COLLAPSE_MATRIX_W3(m00, m01, m02, blendMatrices,  pBlendIndex, pBlendWeight);         \
            break;                                                              \
                                                                                \
        case 4:                                                                 \
            __COLLAPSE_MATRIX_W4(m00, m01, m02, blendMatrices,  pBlendIndex, pBlendWeight);         \
            break;                                                              \
        }                                                                       \
    }


void CMatrixMulVector::ExecuteNativeSSE()
{
#if  VBO_MAP
	glBindBuffer( GL_ARRAY_BUFFER, vertexObj[0] );
	float* pDestPos = (float*)glMapBuffer( GL_ARRAY_BUFFER, GL_READ_WRITE );
#else
	float *pDestPos  = _vertexesDynamic.pVertex;
#endif

	float *pBlendWeight = _vertexesStatic.pWeight;
	unsigned short* pBlendIndex = _vertexesStatic.pIndex;
	float *blendMatrices =  _joints.pMatrix;

	__m128 *pSrcPos = (__m128*)_vertexesStatic.pVertex;

	int srcPosStride, destPosStride;
	srcPosStride = destPosStride = sizeof(float)*4;
	int blendWeightStride, blendIndexStride;
	blendWeightStride = sizeof(float) * SIZE_PER_BONE;
	blendIndexStride = sizeof(unsigned short) * SIZE_PER_BONE;
#if 1
//#pragma omp parallel for
	for(int i=0;i<_vertexesStatic.nSize;i++){
		// 读取操作数：顶点对应的矩阵

		__m128 m00, m01, m02;
		_collapseOneMatrix(
			m00, m01, m02,
			pBlendWeight, pBlendIndex,
			blendMatrices,
			0, 0,
			SIZE_PER_BONE);

		// Advance blend weight and index pointers
		advanceRawPointer(pBlendWeight, blendWeightStride );
		advanceRawPointer(pBlendIndex, blendIndexStride );

		//------------------------------------------------------------------

		// Rearrange to column-major matrix with rows shuffled order to: Z 0 X Y
		__m128 m03 = _mm_setzero_ps();
		__MM_TRANSPOSE4x4_PS(m02, m03, m00, m01);

		// Load source position
		__m128 vI0, vI1, vI2;
		vI0 = __MM_SELECT(*pSrcPos, 0);  //_mm_load_ps1( &pSrcPos->s[0] );
		vI1 = __MM_SELECT(*pSrcPos, 1);  // _mm_load_ps1( &pSrcPos->s[1] );
		vI2 = __MM_SELECT(*pSrcPos, 2);  //_mm_load_ps1( &pSrcPos->s[2] );

		// Transform by collapsed matrix
		__m128 vO = __MM_DOT4x3_PS(m02, m03, m00, m01, vI0, vI1, vI2);   // z 0 x y

		// Store blended position, no aligned requirement
		_mm_storeh_pi((__m64*)(&pDestPos[0]) , vO);
		_mm_store_ss(&pDestPos[2], vO);

		advanceRawPointer(pSrcPos, srcPosStride);
		advanceRawPointer(pDestPos, destPosStride);
	}
#endif

#if  VBO_MAP
	glUnmapBuffer( GL_ARRAY_BUFFER );
	glBindBuffer( GL_ARRAY_BUFFER, NULL );
#endif
}

void CMatrixMulVector::ExecuteNativeSSEOMP()
{
#if  VBO_MAP
	glBindBuffer( GL_ARRAY_BUFFER, vertexObj[0] );
	float* pDestPos = (float*)glMapBuffer( GL_ARRAY_BUFFER, GL_READ_WRITE );
#else
	float *pDestPos  = _vertexesDynamic.pVertex;
#endif

	float *blendMatrices =  _joints.pMatrix;

	float *pSrcPos = _vertexesStatic.pVertex;

#pragma omp parallel for
	for(int i=0;i<_vertexesStatic.nSize;i++){
		// 读取操作数：顶点对应的矩阵

		float *pBlendWeight = _vertexesStatic.pWeight + i*SIZE_PER_BONE;
		cl_ushort* pBlendIndex = _vertexesStatic.pIndex + i*SIZE_PER_BONE;

		__m128 m00, m01, m02;
		_collapseOneMatrix(
			m00, m01, m02,
			pBlendWeight, pBlendIndex,
			blendMatrices,
			0, 0,
			SIZE_PER_BONE);

		//------------------------------------------------------------------

		// Rearrange to column-major matrix with rows shuffled order to: Z 0 X Y
		__m128 m03 = _mm_setzero_ps();
		__MM_TRANSPOSE4x4_PS(m02, m03, m00, m01);

		// Load source position
		__m128 vI0, vI1, vI2;
		vI0 = _mm_load_ps1( &pSrcPos[i+0] );
		vI1 = _mm_load_ps1( &pSrcPos[i+1] );
		vI2 = _mm_load_ps1( &pSrcPos[i+2] );

		// Transform by collapsed matrix
		__m128 vO = __MM_DOT4x3_PS(m02, m03, m00, m01, vI0, vI1, vI2);   // z 0 x y

		// Store blended position, no aligned requirement
		_mm_storeh_pi((__m64*)(&pDestPos[i+0]) , vO);
		_mm_store_ss(&pDestPos[i+2], vO);
	}


#if  VBO_MAP
	glUnmapBuffer( GL_ARRAY_BUFFER );
	glBindBuffer( GL_ARRAY_BUFFER, NULL );
#endif
}

void CMatrixMulVector::ExecuteNativeCPP()
{
#if  VBO_MAP
	glBindBuffer( GL_ARRAY_BUFFER, vertexObj[0] );
	float* pDestPos = (float*)glMapBuffer( GL_ARRAY_BUFFER, GL_READ_WRITE );
#else
	float *pDestPos  = _vertexesDynamicRef.pVertex;
#endif

	float *pBlendWeight = _vertexesStatic.pWeight;
	cl_ushort* pBlendIndex = _vertexesStatic.pIndex;
	float *blendMatrices =  _joints.pMatrix;

	float *pSrcPos = _vertexesStatic.pVertex;


	float sourceVec[3], accumVecPos[3];

	int srcPosStride, destPosStride;
	srcPosStride = destPosStride = sizeof(float)*4;
	int blendWeightStride, blendIndexStride;
	blendWeightStride = sizeof(float) * SIZE_PER_BONE;
	blendIndexStride = sizeof(unsigned short) * SIZE_PER_BONE;

#if USE_OPENMP//use_openmp
#pragma omp parallel for
#endif
	for(int i=0;i<_vertexesStatic.nSize;i++)
	{
		// Load source vertex elements
		for(int j=0;j<3;j++)
		{
			sourceVec[j] = pSrcPos[j];
			accumVecPos[j] = 0.0f ;
		}
	
		// Loop per blend weight
		//
		// Note: Don't change "unsigned short" here!!! If use "size_t" instead,
		// VC7.1 unroll this loop to four blend weights pre-iteration, and then
		// loss performance 10% in this function. Ok, this give a hint that we
		// should unroll this loop manually for better performance, will do that
		// later.
		//
		for (unsigned short blendIdx = 0; blendIdx < SIZE_PER_BONE; ++blendIdx)
		{
			// Blend by multiplying source by blend matrix and scaling by weight
			// Add to accumulator
			// NB weights must be normalised!!
			float weight = 1.0f;
			switch( SIZE_PER_BONE )
			{
			case 2:
				blendIdx ? weight = pBlendWeight[ 1 ] : weight = 1 - pBlendWeight[ 1 ];
				break;

			case 3:
			case 4:
				weight = pBlendWeight[ blendIdx ];
				break;
			default:
				break;
			}

			if (weight)
			{
				// Blend position, use 3x4 matrix
				const float* mat = blendMatrices + pBlendIndex[blendIdx]*MATRIX_SIZE_LINE;
				accumVecPos[0] +=
					(mat[0*4+0] * sourceVec[0] +
					mat[0*4+1] * sourceVec[1] +
					mat[0*4+2] * sourceVec[2] +
					mat[0*4+3])
					* weight;
				accumVecPos[1] +=
					(mat[1*4+0] * sourceVec[0] +
					mat[1*4+1] * sourceVec[1] +
					mat[1*4+2] * sourceVec[2] +
					mat[1*4+3])
					* weight;
				accumVecPos[2] +=
					(mat[2*4+0] * sourceVec[0] +
					mat[2*4+1] * sourceVec[1] +
					mat[2*4+2] * sourceVec[2] +
					mat[2*4+3])
					* weight;
			}
		}
		pDestPos[0] = accumVecPos[0];
		pDestPos[1] = accumVecPos[1];
		pDestPos[2] = accumVecPos[2];

		advanceRawPointer(pSrcPos, srcPosStride);
		advanceRawPointer(pDestPos, destPosStride);
		advanceRawPointer(pBlendWeight, blendWeightStride);
		advanceRawPointer(pBlendIndex, blendIndexStride);
	}

#if  VBO_MAP
	glUnmapBuffer( GL_ARRAY_BUFFER );
	glBindBuffer( GL_ARRAY_BUFFER, NULL );
#endif
}


void CMatrixMulVector::ExecuteNativeCPPOMP()
{
#if  VBO_MAP
	glBindBuffer( GL_ARRAY_BUFFER, vertexObj[0] );
	float* pDestPos = (float*)glMapBuffer( GL_ARRAY_BUFFER, GL_READ_WRITE );
#else
	float *pDestPos  = _vertexesDynamic.pVertex;
#endif

	float *pBlendWeight = _vertexesStatic.pWeight;
	cl_ushort* pBlendIndex = _vertexesStatic.pIndex;
	float *blendMatrices =  _joints.pMatrix;

	float *pSrcPos = _vertexesStatic.pVertex;


#pragma omp parallel for
	for(int i=0;i<_vertexesStatic.nSize;i++)
	{
		float sourceVec[3];
		float accumVecPos[3];

		// Load source vertex elements
		for(int j=0;j<3;j++)
		{
			sourceVec[j] = pSrcPos[j+4*i];
			accumVecPos[j] = 0.0f ;
		}

		// Load accumulators

		// Loop per blend weight
		//
		// Note: Don't change "unsigned short" here!!! If use "size_t" instead,
		// VC7.1 unroll this loop to four blend weights pre-iteration, and then
		// loss performance 10% in this function. Ok, this give a hint that we
		// should unroll this loop manually for better performance, will do that
		// later.
		//
		for (unsigned short blendIdx = 0; blendIdx < SIZE_PER_BONE; ++blendIdx)
		{
			// Blend by multiplying source by blend matrix and scaling by weight
			// Add to accumulator
			// NB weights must be normalised!!
			//float weight = pBlendWeight[i*SIZE_PER_BONE +blendIdx];
			float weight = 1.0f;
			switch( SIZE_PER_BONE )
			{
			case 2:
				blendIdx ? weight = pBlendWeight[ i*SIZE_PER_BONE +1 ] : weight = 1 - pBlendWeight[ i*SIZE_PER_BONE +1 ];
				break;

			case 3:
			case 4:
				weight = pBlendWeight[ i*SIZE_PER_BONE +blendIdx ];
				break;
			default:
				break;
			}

			if (weight)
			{
				// Blend position, use 3x4 matrix
				const float* mat = blendMatrices + pBlendIndex[i*SIZE_PER_BONE +blendIdx]*MATRIX_SIZE_LINE;
				accumVecPos[0] +=
					(mat[0*4+0] * sourceVec[0] +
					mat[0*4+1] * sourceVec[1] +
					mat[0*4+2] * sourceVec[2] +
					mat[0*4+3])
					* weight;
				accumVecPos[1] +=
					(mat[1*4+0] * sourceVec[0] +
					mat[1*4+1] * sourceVec[1] +
					mat[1*4+2] * sourceVec[2] +
					mat[1*4+3])
					* weight;
				accumVecPos[2] +=
					(mat[2*4+0] * sourceVec[0] +
					mat[2*4+1] * sourceVec[1] +
					mat[2*4+2] * sourceVec[2] +
					mat[2*4+3])
					* weight;
			}
		}
		pDestPos[0+4*i] = accumVecPos[0];
		pDestPos[1+4*i] = accumVecPos[1];
		pDestPos[2+4*i] = accumVecPos[2];
	}

#if  VBO_MAP
	glUnmapBuffer( GL_ARRAY_BUFFER );
	glBindBuffer( GL_ARRAY_BUFFER, NULL );
#endif
}

bool CMatrixMulVector::verifyEqual( cl_float4 *v, cl_float4* vRef, int size )
{
	for(int i=0;i<size;i++)
	{
		for (int j=0;j<3;j++)
		{
			float f1=fabs(v[i].s[j] - vRef[i].s[j]);
			float f2=fabs(v[i].s[j]);
			if (  f1/f2  >1e-3 )
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

	//release g_kernel, g_program, and memory objects
	if( g_pfInputBuffer ) {clReleaseMemObject( g_pfInputBuffer ); g_pfInputBuffer = NULL;}
	if( g_pfOCLOutputBuffer ) {clReleaseMemObject( g_pfOCLOutputBuffer ); g_pfOCLOutputBuffer = NULL;}
	if( g_pfOCLIndex ) {clReleaseMemObject( g_pfOCLIndex ); g_pfOCLIndex = NULL;}
	if( g_pfOCLMatrix ) {clReleaseMemObject( g_pfOCLMatrix ); g_pfOCLMatrix = NULL;}

	for (int i=0;i< SIZE_VBO;i++)
	{
		glBindBuffer(1, vertexObj[i]);
		glDeleteBuffers(1, &vertexObj[i]);
	}
}

void CMatrixMulVector::initialize(int sizeVertex, int sizeJoint, streamsdk::SDKCommon * pSampleCommon, TimerList* pTL)
{
	sampleCommon = pSampleCommon;
	_timeValueList = pTL;

	srand(2011);

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

void CMatrixMulVector::SetupKernelVBO(cl_context	pContext, cl_device_id pDevice_ID, cl_kernel pKernel, cl_command_queue pCmdQueue,
																int* nLocationAttrib)
{
	_context = pContext;
	_device_ID = pDevice_ID;
	_kernel = pKernel;
	_cmd_queue = pCmdQueue;

	_locationAttrib = nLocationAttrib;

	glGenVertexArrays( 1, &vertexVAO );
	glBindVertexArray( vertexVAO );

	glGenBuffers( SIZE_VBO, &vertexObj[0]);
	
	// Create Vertex buffer object: 顶点属性 坐标
	glBindBuffer(GL_ARRAY_BUFFER, vertexObj[0]);
	glBufferData(GL_ARRAY_BUFFER, _vertexesStatic.nSize * sizeof(cl_float4), (GLvoid *)_vertexesStatic.pVertex, GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray( 0 );
	glVertexAttribPointer( 0, 4, GL_FLOAT, GL_FALSE, 0, NULL);

	// 	Create Vertex buffer object: 顶点属性 矩阵索引
	glBindBuffer(GL_ARRAY_BUFFER, vertexObj[1]);
	glBufferData(GL_ARRAY_BUFFER, _vertexesStatic.nSize * sizeof(cl_ushort)* SIZE_PER_BONE, (GLvoid *)_vertexesStatic.pIndex, GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray( _locationAttrib[0] );
	glVertexAttribPointer( _locationAttrib[0], 4, GL_FLOAT, GL_FALSE, 0, NULL);

	// 	Create Vertex buffer object: 顶点属性 矩阵权重
	glBindBuffer(GL_ARRAY_BUFFER, vertexObj[2]);
	glBufferData(GL_ARRAY_BUFFER, _vertexesStatic.nSize * sizeof(cl_float)* SIZE_PER_BONE, (GLvoid *)_vertexesStatic.pWeight, GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray( _locationAttrib[1] );
	glVertexAttribPointer( _locationAttrib[1], 4, GL_FLOAT, GL_FALSE, 0, NULL);

	// create OpenCL buffer from GL VBO
	cl_int status = CL_SUCCESS;
	g_pfOCLOutputBuffer = clCreateFromGLBuffer(_context, CL_MEM_WRITE_ONLY, vertexObj[0], NULL);

	const cl_mem_flags INFlags  = CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY; 
	const cl_mem_flags OUTFlags = CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE;
	cl_int errcode_ret;
	g_pfInputBuffer = clCreateBuffer(_context, INFlags,  _vertexesStatic.nSize * sizeof(cl_float4) , _vertexesStatic.pVertex, &errcode_ret);
	g_pfOCLMatrix = clCreateBuffer(_context, INFlags, sizeof(cl_float4) * MATRIX_SIZE_LINE* _joints.nSize , _joints.pMatrix, NULL);

	g_pfOCLIndex = clCreateBuffer(_context, INFlags, sizeof(cl_ushort)* _vertexesStatic.nSize * SIZE_PER_BONE , _vertexesStatic.pIndex, NULL);   
	g_pfOCLWeight = clCreateBuffer(_context, INFlags, sizeof(cl_float)* _vertexesStatic.nSize * SIZE_PER_BONE , _vertexesStatic.pWeight, NULL);   

	//Set kernel arguments
	cl_kernel	kernel = _kernel;
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &g_pfInputBuffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &g_pfOCLIndex);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &g_pfOCLMatrix);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &g_pfOCLOutputBuffer);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &g_pfOCLWeight);

	clSetKernelArg(kernel, 5, sizeof(cl_float4) * MATRIX_SIZE_LINE* _joints.nSize, NULL);
	
	clSetKernelArg(kernel, 6, sizeof(int) , &_vertexesStatic.nSize);

	SetupWorksize();
}
void CMatrixMulVector::SetupKernel(cl_context	pContext, cl_device_id pDevice_ID, cl_kernel pKernel, cl_command_queue pCmdQueue)
{
	_context = pContext;
	_device_ID = pDevice_ID;
	_kernel = pKernel;
	_cmd_queue = pCmdQueue;

	const cl_mem_flags INFlags  = CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY; 
	const cl_mem_flags OUTFlags = CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE;
	cl_int errcode_ret;
	// allocate buffers

	g_pfInputBuffer = clCreateBuffer(_context, INFlags,  _vertexesStatic.nSize * sizeof(cl_float4) , _vertexesStatic.pVertex, &errcode_ret);
	g_pfOCLMatrix = clCreateBuffer(_context, INFlags, sizeof(cl_float4) * MATRIX_SIZE_LINE* _joints.nSize , _joints.pMatrix, NULL);
	g_pfOCLOutputBuffer = clCreateBuffer(_context, OUTFlags, sizeof(cl_float4) *  _vertexesStatic.nSize , _vertexesDynamic.pVertex, NULL);

	g_pfOCLIndex = clCreateBuffer(_context, INFlags, sizeof(cl_ushort)* _vertexesStatic.nSize * SIZE_PER_BONE , _vertexesStatic.pIndex, NULL);   
	g_pfOCLWeight = clCreateBuffer(_context, INFlags, sizeof(cl_float)* _vertexesStatic.nSize * SIZE_PER_BONE , _vertexesStatic.pWeight, NULL);   

	//Set kernel arguments
	cl_kernel	kernel = _kernel;
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &g_pfInputBuffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &g_pfOCLIndex);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &g_pfOCLMatrix);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &g_pfOCLOutputBuffer);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &g_pfOCLWeight);

	SetupWorksize();
}

void CMatrixMulVector::SetupWorksize( )
{
	globalWorkSize[0] = (size_t)sqrtf(_vertexesStatic.nSize);
	globalWorkSize[1] = globalWorkSize[0];
#if VECTOR_FLOAT4
		globalWorkSize[0]/=4; //since proccesing in quadruples
#endif

	localWorkSize[0] = LocalWorkX;
	localWorkSize[1] = LocalWorkX;
	printf("Original global work size (%lu, %lu)\n", globalWorkSize[0], globalWorkSize[1]);
	printf("Original local work size (%lu, %lu)\n", localWorkSize[0], localWorkSize[1]);

	size_t  workGroupSizeMaximum;
	clGetKernelWorkGroupInfo(_kernel, _device_ID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void *)&workGroupSizeMaximum, NULL);
	printf("Maximum workgroup size for this kernel  %lu\n\n",workGroupSizeMaximum );

	if ( _vertexesStatic.nSize>workGroupSizeMaximum )
	{
		globalWorkSize[0] = workGroupSizeMaximum;
		globalWorkSize[1] = _vertexesStatic.nSize / workGroupSizeMaximum;
	}
	printf("Actual global work size (%lu, %lu)\n", globalWorkSize[0], globalWorkSize[1]);
}

bool CMatrixMulVector::ExecuteKernel()
{
	cl_int err = CL_SUCCESS;

	//printf("Executing OpenCL kernel...");

	cl_event g_perf_event = NULL;
	// execute kernel, pls notice g_bAutoGroupSize
	err= clEnqueueNDRangeKernel(_cmd_queue, _kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &g_perf_event);
	if (err != CL_SUCCESS)
	{
		printf("ERROR: Failed to execute kernel...\n");
		return false;
	}
	err = clWaitForEvents(1, &g_perf_event);
	if (err != CL_SUCCESS)
	{
		printf("ERROR: Failed to clWaitForEvents...\n");
		return false;
	}

	//printf("Done\n");


	void* tmp_ptr = NULL;

	

#if !VECTOR_FLOAT4
		err = clEnqueueReadBuffer(_cmd_queue, g_pfOCLOutputBuffer, CL_TRUE, 0, sizeof(cl_float) *VERTEX_VECTOR_SIZE* _vertexesStatic.nSize , _vertexesDynamic.pVertex, 0, NULL, NULL);
#else
		err = clEnqueueReadBuffer(_cmd_queue, g_pfOCLOutputBuffer, CL_TRUE, 0, sizeof(cl_float4) * _vertexesStatic.nSize , _vertexesDynamic.pVertex, 0, NULL, NULL);
#endif
		if (err != CL_SUCCESS)
		{
			printf("ERROR: Failed to clEnqueueReadBuffer...\n");
			return false;
		}
	
	clFinish(_cmd_queue);

	clEnqueueUnmapMemObject(_cmd_queue, g_pfOCLOutputBuffer, tmp_ptr, 0, NULL, NULL);
	
	return true;
}
bool CMatrixMulVector::ExecuteKernelVBO()
{

	// Acquire GL buffer
	clEnqueueAcquireGLObjects(_cmd_queue, 1, &g_pfOCLOutputBuffer, 0, 0, NULL);


	cl_event g_perf_event = NULL;
	clEnqueueNDRangeKernel(_cmd_queue, _kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &g_perf_event);
	clWaitForEvents(1, &g_perf_event);

	static bool bRunOnce = false;
	if ( !bRunOnce)
	{
		clEnqueueReadBuffer(_cmd_queue, g_pfOCLOutputBuffer, CL_TRUE, 0, sizeof(cl_float4) * _vertexesStatic.nSize , _vertexesDynamic.pVertex, 0, NULL, NULL);
		bRunOnce = true;
	}
	

	// Release GL buffer
	clEnqueueReleaseGLObjects(_cmd_queue, 1, &g_pfOCLOutputBuffer, 0, 0, 0);
	
	clFinish(_cmd_queue);

	return true;
}

void CMatrixMulVector::insertTimer( std::string item, double time)
{
	if ( _timeValueList->size()>100 )
	{
		return;
	}
	_timeValueList->insert( std::make_pair(item, time) );
}

void CMatrixMulVector::renderVBO()
{
	// render from the vAo
	glBindVertexArray( vertexVAO );
	
	glDrawArrays(GL_POINTS, 0,  _vertexesStatic.nSize );
	
	glBindVertexArray( NULL );

}

void CMatrixMulVector::ExecuteNativeCPPT1()
{
#if  VBO_MAP
	glBindBuffer( GL_ARRAY_BUFFER, vertexObj[0] );
	float* pDestPos = (float*)glMapBuffer( GL_ARRAY_BUFFER, GL_READ_WRITE );
#else
	float *pDestPos  = _vertexesDynamicRef.pVertex;
#endif

	float *pSrcPos = _vertexesStatic.pVertex;

	for(int i=0;i<_vertexesStatic.nSize*4;i++)
	{
		pDestPos[i] = pSrcPos[i];
	}

#if  VBO_MAP
	glUnmapBuffer( GL_ARRAY_BUFFER );
	glBindBuffer( GL_ARRAY_BUFFER, NULL );
#endif
}

void CMatrixMulVector::ExecuteNativeCPPOMPT1()
{
#if  VBO_MAP
	glBindBuffer( GL_ARRAY_BUFFER, vertexObj[0] );
	float* pDestPos = (float*)glMapBuffer( GL_ARRAY_BUFFER, GL_READ_WRITE );
#else
	float *pDestPos  = _vertexesDynamicRef.pVertex;
#endif

	float *pSrcPos = _vertexesStatic.pVertex;

#pragma omp parallel for
	for(int i=0;i<_vertexesStatic.nSize*4;i++)
	{
		pDestPos[i] = pSrcPos[i];
	}

#if  VBO_MAP
	glUnmapBuffer( GL_ARRAY_BUFFER );
	glBindBuffer( GL_ARRAY_BUFFER, NULL );
#endif
}

void CMatrixMulVector::ExecuteNativeSSET1()
{
#if  VBO_MAP
	glBindBuffer( GL_ARRAY_BUFFER, vertexObj[0] );
	__m128* pDestPos = (__m128*)glMapBuffer( GL_ARRAY_BUFFER, GL_READ_WRITE );
#else
	__m128 *pDestPos  = (__m128*)_vertexesDynamic.pVertex;
#endif

	__m128 *pSrcPos = (__m128*)_vertexesStatic.pVertex;

	for(int i=0;i<_vertexesStatic.nSize;i++){
		*(pDestPos++) = *(pSrcPos++);
	}

#if  VBO_MAP
	glUnmapBuffer( GL_ARRAY_BUFFER );
	glBindBuffer( GL_ARRAY_BUFFER, NULL );
#endif
}

void CMatrixMulVector::ExecuteNativeSSEOMPT1()
{

#if  VBO_MAP
	glBindBuffer( GL_ARRAY_BUFFER, vertexObj[0] );
	__m128* pDestPos = (__m128*)glMapBuffer( GL_ARRAY_BUFFER, GL_READ_WRITE );
#else
	__m128 *pDestPos  = (__m128*)_vertexesDynamic.pVertex;
#endif

	__m128 *pSrcPos = (__m128*)_vertexesStatic.pVertex;

#pragma omp parallel for
	for(int i=0;i<_vertexesStatic.nSize;i++){
		pDestPos[i] = pSrcPos[i];
	}

#if  VBO_MAP
	glUnmapBuffer( GL_ARRAY_BUFFER );
	glBindBuffer( GL_ARRAY_BUFFER, NULL );
#endif
}
