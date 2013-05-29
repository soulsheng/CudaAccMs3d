/**********************************************************************
Copyright ?012 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

?Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
?Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#define  SIZE_PER_BONE   2
#define MATRIX_SIZE_LINE 3
#define    JOINT_SIZE    (1<<6)

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__kernel
void sineWave(
    __global float4 * pos,
    unsigned int width,
    unsigned int height,
    float time)
{
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sin(u*freq + time) * cos(v*freq + time) * 0.5f;

    // write output vertex
    pos[y*width+x] = (float4)(u, w, v, 1.0f);
}


__kernel void
updateVectorByMatrix4( const __global float4 *pInput, const __global ushort *pIndex, __constant  float4 *pMatrix,__global float4 *pOutput
						,  const __global float *pWeight, __local float4* pMatrixShared, int nSize)
{

	size_t threadIndex = get_global_id(0) + get_global_id(1) *get_global_size(0);
	float4 sourceVec = pInput[threadIndex], accumVecPos;

		// Load accumulators
		accumVecPos = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

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
				if( !blendIdx )
					weight = 1.0f - pWeight[ 1+ SIZE_PER_BONE*threadIndex ];
				else
					weight = pWeight[ 1+ SIZE_PER_BONE*threadIndex ];
				break;

			case 3:
			case 4:
				weight = pWeight[ blendIdx + SIZE_PER_BONE*threadIndex ];
				break;
			default:
				break;
			}

			if (weight)
			{
				// Blend position, use 3x4 matrix
				ushort matrixIndex = pIndex[blendIdx + SIZE_PER_BONE*threadIndex]*MATRIX_SIZE_LINE;
				accumVecPos.x +=
					(pMatrix[matrixIndex+0].x * sourceVec.x +
					pMatrix[matrixIndex+0].y * sourceVec.y +
					pMatrix[matrixIndex+0].z * sourceVec.z +
					pMatrix[matrixIndex+0].w)
					* weight;
				accumVecPos.y +=
					(pMatrix[matrixIndex+1].x * sourceVec.x +
					pMatrix[matrixIndex+1].y * sourceVec.y +
					pMatrix[matrixIndex+1].z * sourceVec.z +
					pMatrix[matrixIndex+1].w)
					* weight;
				accumVecPos.z +=
					(pMatrix[matrixIndex+2].x * sourceVec.x +
					pMatrix[matrixIndex+2].y * sourceVec.y +
					pMatrix[matrixIndex+2].z * sourceVec.z +
					pMatrix[matrixIndex+2].w)
					* weight;
			}
		}
		pOutput[ threadIndex ] = accumVecPos;

}


__kernel void
updateVectorByMatrix4Shared( const __global float4 *pInput, const __global ushort *pIndex, __constant  float4 *pMatrix,__global float4 *pOutput
						,  const __global float *pWeight, __local float4* pMatrixShared, int nSize)
{	

	size_t threadIndex = get_global_id(0) + get_global_id(1) *get_global_size(0);

	size_t localIndex = get_local_id(0) +  get_local_id(1) *get_local_size(0);
	if( localIndex < JOINT_SIZE )
	{
		pMatrixShared[ localIndex*MATRIX_SIZE_LINE ] = pMatrix[ localIndex*MATRIX_SIZE_LINE ];
		pMatrixShared[ localIndex*MATRIX_SIZE_LINE+1 ] = pMatrix[ localIndex*MATRIX_SIZE_LINE+1 ];
		pMatrixShared[ localIndex*MATRIX_SIZE_LINE+2 ] = pMatrix[ localIndex*MATRIX_SIZE_LINE+2 ];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 sourceVec = pInput[threadIndex], accumVecPos;

		// Load accumulators
		accumVecPos = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

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
				if( !blendIdx )
					weight = 1.0f - pWeight[ 1+ SIZE_PER_BONE*threadIndex ];
				else
					weight = pWeight[ 1+ SIZE_PER_BONE*threadIndex ];
				break;

			case 3:
			case 4:
				weight = pWeight[ blendIdx + SIZE_PER_BONE*threadIndex ];
				break;
			default:
				break;
			}

			if (weight)
			{
				// Blend position, use 3x4 matrix
				ushort matrixIndex = pIndex[blendIdx + SIZE_PER_BONE*threadIndex]*MATRIX_SIZE_LINE;
				accumVecPos.x +=
					(pMatrixShared[matrixIndex+0].x * sourceVec.x +
					pMatrixShared[matrixIndex+0].y * sourceVec.y +
					pMatrixShared[matrixIndex+0].z * sourceVec.z +
					pMatrixShared[matrixIndex+0].w)
					* weight;
				accumVecPos.y +=
					(pMatrixShared[matrixIndex+1].x * sourceVec.x +
					pMatrixShared[matrixIndex+1].y * sourceVec.y +
					pMatrixShared[matrixIndex+1].z * sourceVec.z +
					pMatrixShared[matrixIndex+1].w)
					* weight;
				accumVecPos.z +=
					(pMatrixShared[matrixIndex+2].x * sourceVec.x +
					pMatrixShared[matrixIndex+2].y * sourceVec.y +
					pMatrixShared[matrixIndex+2].z * sourceVec.z +
					pMatrixShared[matrixIndex+2].w)
					* weight;
			}
		}
		pOutput[ threadIndex ] = accumVecPos;

}


__kernel void
updateVectorByMatrix4SharedCoalesce( const __global float4 *pInput, const __global ushort *pIndex, __constant  float4 *pMatrix,__global float4 *pOutput
						,  const __global float *pWeight, __local float4* pMatrixShared, int nSize)
{	

	size_t threadIndex = get_global_id(0) + get_global_id(1) *get_global_size(0);

	size_t localIndex = get_local_id(0) +  get_local_id(1) *get_local_size(0);
	if( localIndex < JOINT_SIZE )
	{
		pMatrixShared[ localIndex*MATRIX_SIZE_LINE ] = pMatrix[ localIndex*MATRIX_SIZE_LINE ];
		pMatrixShared[ localIndex*MATRIX_SIZE_LINE+1 ] = pMatrix[ localIndex*MATRIX_SIZE_LINE+1 ];
		pMatrixShared[ localIndex*MATRIX_SIZE_LINE+2 ] = pMatrix[ localIndex*MATRIX_SIZE_LINE+2 ];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 sourceVec = pInput[threadIndex], accumVecPos;

		// Load accumulators
		accumVecPos = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

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
				if( !blendIdx )
					weight = 1.0f - pWeight[ 1*nSize+ threadIndex ];
				else
					weight = pWeight[ 1*nSize+ threadIndex ];
				break;

			case 3:
			case 4:
				weight = pWeight[ blendIdx*nSize + threadIndex ];
				break;
			default:
				break;
			}

			if (weight)
			{
				// Blend position, use 3x4 matrix
				ushort matrixIndex = pIndex[blendIdx*nSize + threadIndex]*MATRIX_SIZE_LINE;
				accumVecPos.x +=
					(pMatrixShared[matrixIndex+0].x * sourceVec.x +
					pMatrixShared[matrixIndex+0].y * sourceVec.y +
					pMatrixShared[matrixIndex+0].z * sourceVec.z +
					pMatrixShared[matrixIndex+0].w)
					* weight;
				accumVecPos.y +=
					(pMatrixShared[matrixIndex+1].x * sourceVec.x +
					pMatrixShared[matrixIndex+1].y * sourceVec.y +
					pMatrixShared[matrixIndex+1].z * sourceVec.z +
					pMatrixShared[matrixIndex+1].w)
					* weight;
				accumVecPos.z +=
					(pMatrixShared[matrixIndex+2].x * sourceVec.x +
					pMatrixShared[matrixIndex+2].y * sourceVec.y +
					pMatrixShared[matrixIndex+2].z * sourceVec.z +
					pMatrixShared[matrixIndex+2].w)
					* weight;
			}
		}
		pOutput[ threadIndex ] = accumVecPos;

}

__kernel void
updateVectorByMatrix4Ideal1( const __global float4 *pInput, const __global ushort *pIndex, __constant  float4 *pMatrix,__global float4 *pOutput
						,  const __global float *pWeight, __local float4* pMatrixShared, int nSize)
{

	size_t threadIndex = get_global_id(0) + get_global_id(1) *get_global_size(0);
	
	float4 sourceVec = pInput[threadIndex];

	pOutput[ threadIndex ] = sourceVec;

}

__kernel void
updateVectorByMatrix4Ideal2( const __global float4 *pInput, const __global ushort *pIndex, __constant  float4 *pMatrix,__global float4 *pOutput
						,  const __global float *pWeight, __local float4* pMatrixShared, int nSize)
{

	size_t threadIndex = get_global_id(0) + get_global_id(1) *get_global_size(0);
	float4 sourceVec = pInput[threadIndex], accumVecPos;
	
	accumVecPos = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

	ushort matrixIndex = pIndex[threadIndex]*MATRIX_SIZE_LINE;

				accumVecPos.x +=
					(pMatrix[matrixIndex+0].x * sourceVec.x +
					pMatrix[matrixIndex+0].y * sourceVec.y +
					pMatrix[matrixIndex+0].z * sourceVec.z +
					pMatrix[matrixIndex+0].w);
				accumVecPos.y +=
					(pMatrix[matrixIndex+1].x * sourceVec.x +
					pMatrix[matrixIndex+1].y * sourceVec.y +
					pMatrix[matrixIndex+1].z * sourceVec.z +
					pMatrix[matrixIndex+1].w) ;
				accumVecPos.z +=
					(pMatrix[matrixIndex+2].x * sourceVec.x +
					pMatrix[matrixIndex+2].y * sourceVec.y +
					pMatrix[matrixIndex+2].z * sourceVec.z +
					pMatrix[matrixIndex+2].w) ;

	pOutput[ threadIndex ] = accumVecPos;

}


__kernel void
updateVectorByMatrix4TwoWeight( const __global float4 *pInput, const __global ushort *pIndex, __constant  float4 *pMatrix,__global float4 *pOutput
						,  const __global float *pWeight, __local float4* pMatrixShared, int nSize)
{

	size_t threadIndex = get_global_id(0) + get_global_id(1) *get_global_size(0);
	float4 sourceVec = pInput[threadIndex];
	sourceVec.w = 1.0f;

		ushort matrixIndex1 = pIndex[SIZE_PER_BONE*threadIndex]*MATRIX_SIZE_LINE;
		ushort matrixIndex2 = pIndex[SIZE_PER_BONE*threadIndex +1]*MATRIX_SIZE_LINE;
		float weight = pWeight[ 1+ SIZE_PER_BONE*threadIndex ];

		float4 weight1 = (float4)(1-weight, 1-weight, 1-weight, 1-weight);
		float4 weight2 = (float4)(weight, weight, weight, weight);

		float4 mat0 = pMatrix[matrixIndex1+0] * weight1 + pMatrix[matrixIndex2+0] * weight2;
		float4  xResult = mat0 * sourceVec;

		float4 mat1 = pMatrix[matrixIndex1+1] * weight1 + pMatrix[matrixIndex2+1] * weight2;
		float4  yResult = mat1 * sourceVec;
		
		float4 mat2 = pMatrix[matrixIndex1+2] * weight1 + pMatrix[matrixIndex2+2] * weight2;
		float4  zResult = mat2 * sourceVec;

		float4 xResultT = (float4)(xResult.x, yResult.x, zResult.x, 0.0f);
		float4 yResultT = (float4)(xResult.y, yResult.y, zResult.y, 0.0f);
		float4 zResultT = (float4)(xResult.z, yResult.z, zResult.z, 0.0f);
		float4 wResultT = (float4)(xResult.w, yResult.w, zResult.w, 0.0f);

		pOutput[threadIndex] = xResultT + yResultT + zResultT + wResultT;
		/*
		pOutput[ threadIndex ].x =
					(pMatrix[matrixIndex1+0].x * (1-weight) + pMatrix[matrixIndex2+0].x * weight  )* sourceVec.x +
					(pMatrix[matrixIndex1+0].y * (1-weight) + pMatrix[matrixIndex2+0].y * weight  ) * sourceVec.y +
					(pMatrix[matrixIndex1+0].z * (1-weight) + pMatrix[matrixIndex2+0].z * weight  ) * sourceVec.z +
					(pMatrix[matrixIndex1+0].w * (1-weight) + pMatrix[matrixIndex2+0].w * weight  );

		pOutput[ threadIndex ].y =
					(pMatrix[matrixIndex1+1].x * (1-weight) + pMatrix[matrixIndex2+1].x * weight  ) * sourceVec.x +
					(pMatrix[matrixIndex1+1].y * (1-weight) + pMatrix[matrixIndex2+1].y * weight  ) * sourceVec.y +
					(pMatrix[matrixIndex1+1].z * (1-weight) + pMatrix[matrixIndex2+1].z * weight  ) * sourceVec.z +
					(pMatrix[matrixIndex1+1].w * (1-weight) + pMatrix[matrixIndex2+1].w * weight  );

		pOutput[ threadIndex ].z =
					(pMatrix[matrixIndex1+2].x * (1-weight) + pMatrix[matrixIndex2+2].x * weight  ) * sourceVec.x +
					(pMatrix[matrixIndex1+2].y * (1-weight) + pMatrix[matrixIndex2+2].y * weight  ) * sourceVec.y +
					(pMatrix[matrixIndex1+2].z * (1-weight) + pMatrix[matrixIndex2+2].z * weight  ) * sourceVec.z +
					(pMatrix[matrixIndex1+2].w * (1-weight) + pMatrix[matrixIndex2+2].w * weight  );
		*/
}


__kernel void
updateVectorByMatrix4OneWeight( const __global float4 *pInput, const __global ushort *pIndex, __constant  float4 *pMatrix,__global float4 *pOutput
						,  const __global float *pWeight, __local float4* pMatrixShared, int nSize)
{

	size_t threadIndex = get_global_id(0) + get_global_id(1) *get_global_size(0);
	float4 sourceVec = pInput[threadIndex];
	float weight = sourceVec.w;
	sourceVec.w = 1.0f;

		ushort matrixIndex1 = pIndex[SIZE_PER_BONE*threadIndex]*MATRIX_SIZE_LINE;

		float4 weight1 = (float4)(weight, weight, weight, weight);

		float4 mat0 = pMatrix[matrixIndex1+0] * weight1 ;
		float4  xResult = mat0 * sourceVec;

		float4 mat1 = pMatrix[matrixIndex1+1] * weight1 ;
		float4  yResult = mat1 * sourceVec;
		
		float4 mat2 = pMatrix[matrixIndex1+2] * weight1 ;
		float4  zResult = mat2 * sourceVec;

		float4 xResultT = (float4)(xResult.x, yResult.x, zResult.x, 0.0f);
		float4 yResultT = (float4)(xResult.y, yResult.y, zResult.y, 0.0f);
		float4 zResultT = (float4)(xResult.z, yResult.z, zResult.z, 0.0f);
		float4 wResultT = (float4)(xResult.w, yResult.w, zResult.w, 0.0f);

		pOutput[threadIndex] = xResultT + yResultT + zResultT + wResultT;
}

__kernel void
updateVectorByMatrix4MultiWeight( const __global float4 *pInput, const __global ushort *pIndex, __constant  float4 *pMatrix,__global float4 *pOutput
						,  const __global float *pWeight, __local float4* pMatrixShared, int nSize)
{
	size_t threadIndex = get_global_id(0) + get_global_id(1) *get_global_size(0);
	float4 sourceVec = pInput[threadIndex];

	ushort matrixIndex1 = pIndex[SIZE_PER_BONE*threadIndex ]*MATRIX_SIZE_LINE;
	float weight1 = pWeight[ SIZE_PER_BONE*threadIndex ];
	float4 weight11 = (float4)(weight1, weight1, weight1, weight1);

	float4 mat[3] ;
	mat[0] = pMatrix[matrixIndex1+0] * weight11; 
	mat[1] = pMatrix[matrixIndex1+1] * weight11; 
	mat[2] = pMatrix[matrixIndex1+2] * weight11; 
	

	for(int i=1; i<SIZE_PER_BONE; i++)
	{
		matrixIndex1 = pIndex[SIZE_PER_BONE*threadIndex + i]*MATRIX_SIZE_LINE;
		weight1 = pWeight[ SIZE_PER_BONE*threadIndex + i ];
		weight11 = (float4)(weight1, weight1, weight1, weight1);

		mat[0] += pMatrix[matrixIndex1+0] * weight11; 
		mat[1] += pMatrix[matrixIndex1+1] * weight11; 
		mat[2] += pMatrix[matrixIndex1+2] * weight11; 
	}
	
	float4  xResult = mat[0] * sourceVec;
	float4  yResult = mat[1] * sourceVec;
	float4  zResult = mat[2] * sourceVec;

	float4 xResultT = (float4)(xResult.x, yResult.x, zResult.x, 0.0f);
	float4 yResultT = (float4)(xResult.y, yResult.y, zResult.y, 0.0f);
	float4 zResultT = (float4)(xResult.z, yResult.z, zResult.z, 0.0f);
	float4 wResultT = (float4)(xResult.w, yResult.w, zResult.w, 0.0f);
	pOutput[threadIndex] = xResultT + yResultT + zResultT + wResultT;
}

__kernel void
updateVectorByMatrix4MultiWeightShared( const __global float4 *pInput, const __global ushort *pIndex, __constant  float4 *pMatrix,__global float4 *pOutput
						,  const __global float *pWeight, __local float4* pMatrixShared, int nSize)
{
#if 1
	size_t threadIndex = get_global_id(0) + get_global_id(1) *get_global_size(0);

	size_t localIndex = get_local_id(0) +  get_local_id(1) *get_local_size(0);
	if( localIndex < JOINT_SIZE )
	{
		pMatrixShared[ localIndex*MATRIX_SIZE_LINE ] = pMatrix[ localIndex*MATRIX_SIZE_LINE ];
		pMatrixShared[ localIndex*MATRIX_SIZE_LINE+1 ] = pMatrix[ localIndex*MATRIX_SIZE_LINE+1 ];
		pMatrixShared[ localIndex*MATRIX_SIZE_LINE+2 ] = pMatrix[ localIndex*MATRIX_SIZE_LINE+2 ];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 sourceVec = pInput[threadIndex];

	ushort matrixIndex1 = pIndex[SIZE_PER_BONE*threadIndex ]*MATRIX_SIZE_LINE;
	float weight1 = pWeight[ SIZE_PER_BONE*threadIndex ];
	float4 weight11 = (float4)(weight1, weight1, weight1, weight1);

	float4 mat[3] ;
	mat[0] = pMatrixShared[matrixIndex1+0] * weight11; 
	mat[1] = pMatrixShared[matrixIndex1+1] * weight11; 
	mat[2] = pMatrixShared[matrixIndex1+2] * weight11; 
	

	for(int i=1; i<SIZE_PER_BONE; i++)
	{
		matrixIndex1 = pIndex[SIZE_PER_BONE*threadIndex + i]*MATRIX_SIZE_LINE;
		weight1 = pWeight[ SIZE_PER_BONE*threadIndex + i ];
		weight11 = (float4)(weight1, weight1, weight1, weight1);

		mat[0] += pMatrixShared[matrixIndex1+0] * weight11; 
		mat[1] += pMatrixShared[matrixIndex1+1] * weight11; 
		mat[2] += pMatrixShared[matrixIndex1+2] * weight11; 
	}
	
	float4  xResult = mat[0] * sourceVec;
	float4  yResult = mat[1] * sourceVec;
	float4  zResult = mat[2] * sourceVec;

	float4 xResultT = (float4)(xResult.x, yResult.x, zResult.x, 0.0f);
	float4 yResultT = (float4)(xResult.y, yResult.y, zResult.y, 0.0f);
	float4 zResultT = (float4)(xResult.z, yResult.z, zResult.z, 0.0f);
	float4 wResultT = (float4)(xResult.w, yResult.w, zResult.w, 0.0f);
	pOutput[threadIndex] = xResultT + yResultT + zResultT + wResultT;
#else
	size_t threadIndex = get_global_id(0) + get_global_id(1) *get_global_size(0);
	pOutput[threadIndex] = pInput[threadIndex];
#endif
}