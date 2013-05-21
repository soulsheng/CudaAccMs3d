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

#define  SIZE_PER_BONE   4
#define MATRIX_SIZE_LINE 3
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
updateVectorByMatrix4( const __global float4 *pInput, const __global ushort *pIndex,__global float4 *pMatrix,__global float4 *pOutput
						,  const __global float *pWeight)
{
	/*
	size_t index = get_global_id(0) + get_global_id(1) *get_global_size(0);
	
	int offset = pIndex[index]*3;

	float4 vIn = pInput[index]; 
	
	pOutput[index] = (float4)( 
		vIn.x * pMatrix[offset+0].x + vIn.y * pMatrix[offset+0].y + vIn.z * pMatrix[offset+0].z  + pMatrix[offset+0].w ,
		vIn.x * pMatrix[offset+1].x + vIn.y * pMatrix[offset+1].y + vIn.z * pMatrix[offset+1].z  + pMatrix[offset+1].w ,
		vIn.x * pMatrix[offset+2].x + vIn.y * pMatrix[offset+2].y + vIn.z * pMatrix[offset+2].z  + pMatrix[offset+2].w ,
		1.0f);
		*/
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