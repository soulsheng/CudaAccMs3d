// Copyright (c) 2009-2011 Intel Corporation
// All rights reserved.
// 
// WARRANTY DISCLAIMER
// 
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Intel Corporation is the author of the Materials, and requests that all
// problem reports or change requests be submitted to it directly

__kernel void
SimpleKernel( const __global float *input, __global float *output)
{
	size_t index = get_global_id(0);
    output[index] = rsqrt( input[index] );
}

__kernel /*__attribute__((vec_type_hint(float4))) */ void
SimpleKernel4( const __global float4 *input, __global float4 *output)
{
	size_t index = get_global_id(0);
    output[index] = rsqrt( input[index] );
}

__kernel void
updateVectorByMatrix( const __global float *pInput, const __global int *pIndex,const __global float *pMatrix,__global float *pOutput)
{
	size_t index = get_global_id(0) + get_global_id(1) *get_global_size(0);
	
	const __global float *pMat = pMatrix + pIndex[index]*3*4;

	pOutput[4*index+0] = pInput[4*index] * pMat[0] + pInput[4*index+1] * pMat[1] + pInput[4*index+2] * pMat[2]  + pMat[3];
	pOutput[4*index+1] = pInput[4*index] * pMat[4] + pInput[4*index+1] * pMat[5] + pInput[4*index+2] * pMat[6]  + pMat[7];
	pOutput[4*index+2] = pInput[4*index] * pMat[8] + pInput[4*index+1] * pMat[9] + pInput[4*index+2] * pMat[10]  + pMat[11];
}