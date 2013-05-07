
#include "MatrixMulVector.h"
#include <omp.h>
//#include "stdafx.h"
#include <math.h>
#include <stdio.h>
#include <emmintrin.h>
#include <CL/cl_gl.h>

#define  LocalWorkX		8
#define  LocalWorkY		8

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
#if 1
#pragma omp parallel for
	for(int i=0;i<_vertexesStatic.nSize;i++){
		// 读取操作数：顶点对应的矩阵
		//float *mat = _joints.pMatrix + _vertexesStatic.pIndex[i]*MATRIX_SIZE_LINE*4;
		cl_float4 *pMat =  _joints.pMatrix + _vertexesStatic.pIndex[i]*MATRIX_SIZE_LINE;

		__m128 m0, m1, m2, m3;
		m0 = _mm_load_ps( &pMat[0].s[0] );
		m1 = _mm_load_ps( &pMat[1].s[0] );
		m2 = _mm_load_ps( &pMat[2].s[0] );

		// Rearrange to column-major matrix with rows shuffled order to: Z 0 X Y
		m3 = _mm_setzero_ps();
		__MM_TRANSPOSE4x4_PS(m2, m3, m0, m1);

		// Load source position
		__m128 vI0, vI1, vI2;
		vI0 = _mm_load_ps1( &_vertexesStatic.pVertex[i].s[0] );
		vI1 = _mm_load_ps1( &_vertexesStatic.pVertex[i].s[1] );
		vI2 = _mm_load_ps1( &_vertexesStatic.pVertex[i].s[2] );

		// Transform by collapsed matrix
		__m128 vO = __MM_DOT4x3_PS(m2, m3, m0, m1, vI0, vI1, vI2);   // z 0 x y

		// Store blended position, no aligned requirement
		_mm_storeh_pi((__m64*)(&_vertexesDynamicRef.pVertex[i].s[0]) , vO);
		_mm_store_ss(&_vertexesDynamicRef.pVertex[i].s[2], vO);

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

	glBindBuffer(1, vertexObj);
	glDeleteBuffers(1, &vertexObj);
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

void CMatrixMulVector::SetupKernelVBO(cl_context	pContext, cl_device_id pDevice_ID, cl_kernel pKernel, cl_command_queue pCmdQueue)
{
	_context = pContext;
	_device_ID = pDevice_ID;
	_kernel = pKernel;
	_cmd_queue = pCmdQueue;

	// Create Vertex buffer object
	glGenBuffers(1, &vertexObj);
	glBindBuffer(GL_ARRAY_BUFFER, vertexObj);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, _vertexesStatic.nSize * sizeof(cl_float4), (GLvoid *)_vertexesStatic.pVertex, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// create OpenCL buffer from GL VBO
	cl_int status = CL_SUCCESS;
	g_pfOCLOutputBuffer = clCreateFromGLBuffer(_context, CL_MEM_WRITE_ONLY, vertexObj, NULL);

	const cl_mem_flags INFlags  = CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY; 
	const cl_mem_flags OUTFlags = CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE;
	cl_int errcode_ret;
	g_pfInputBuffer = clCreateBuffer(_context, INFlags,  _vertexesStatic.nSize * sizeof(cl_float4) , _vertexesStatic.pVertex, &errcode_ret);
	g_pfOCLMatrix = clCreateBuffer(_context, INFlags, sizeof(cl_float4) * MATRIX_SIZE_LINE* _joints.nSize , _joints.pMatrix, NULL);
	g_pfOCLIndex = clCreateBuffer(_context, INFlags, sizeof(cl_int)* _vertexesStatic.nSize , _vertexesStatic.pIndex, NULL);   

	//Set kernel arguments
	cl_kernel	kernel = _kernel;
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &g_pfInputBuffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &g_pfOCLIndex);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &g_pfOCLMatrix);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &g_pfOCLOutputBuffer);

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
#if !VECTOR_FLOAT4
	g_pfInputBuffer = clCreateBuffer(_context, INFlags, _vertexesStatic.nSize*VERTEX_VECTOR_SIZE * sizeof(cl_float) , _vertexesStatic.pVertex, &errcode_ret);
	g_pfOCLMatrix = clCreateBuffer(_context, INFlags, sizeof(cl_float)*VERTEX_VECTOR_SIZE * MATRIX_SIZE_LINE*_joints.nSize , _joints.pMatrix, NULL);
	g_pfOCLOutputBuffer = clCreateBuffer(_context, OUTFlags, sizeof(cl_float)*VERTEX_VECTOR_SIZE * _vertexesStatic.nSize , _vertexesDynamic.pVertex, NULL);
#else
	g_pfInputBuffer = clCreateBuffer(_context, INFlags,  _vertexesStatic.nSize * sizeof(cl_float4) , _vertexesStatic.pVertex, &errcode_ret);
	g_pfOCLMatrix = clCreateBuffer(_context, INFlags, sizeof(cl_float4) * MATRIX_SIZE_LINE* _joints.nSize , _joints.pMatrix, NULL);
	g_pfOCLOutputBuffer = clCreateBuffer(_context, OUTFlags, sizeof(cl_float4) *  _vertexesStatic.nSize , _vertexesDynamic.pVertex, NULL);
#endif

	g_pfOCLIndex = clCreateBuffer(_context, INFlags, sizeof(cl_int)* _vertexesStatic.nSize , _vertexesStatic.pIndex, NULL);   

	//Set kernel arguments
	cl_kernel	kernel = _kernel;
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &g_pfInputBuffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &g_pfOCLIndex);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &g_pfOCLMatrix);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &g_pfOCLOutputBuffer);

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
	int timer = createTimer();
	resetTimer(timer);
	startTimer(timer);

	// Acquire GL buffer
	clEnqueueAcquireGLObjects(_cmd_queue, 1, &g_pfOCLOutputBuffer, 0, 0, NULL);
	
	stopTimer(timer);
	double dTime = (cl_double)readTimer(timer);
	insertTimer("11.Acquire GL", dTime);

	resetTimer(timer);
	startTimer(timer);

	cl_event g_perf_event = NULL;
	clEnqueueNDRangeKernel(_cmd_queue, _kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &g_perf_event);
	clWaitForEvents(1, &g_perf_event);

	stopTimer(timer);
	dTime = (cl_double)readTimer(timer);
	insertTimer("12.executingKernel", dTime);

	resetTimer(timer);
	startTimer(timer);

	static bool bRunOnce = false;
	if ( !bRunOnce)
	{
		clEnqueueReadBuffer(_cmd_queue, g_pfOCLOutputBuffer, CL_TRUE, 0, sizeof(cl_float4) * _vertexesStatic.nSize , _vertexesDynamic.pVertex, 0, NULL, NULL);
		bRunOnce = true;
	}
	
	stopTimer(timer);
	dTime = (cl_double)readTimer(timer);
	insertTimer("13.ReadBuffer", dTime);

	resetTimer(timer);
	startTimer(timer);

	// Release GL buffer
	clEnqueueReleaseGLObjects(_cmd_queue, 1, &g_pfOCLOutputBuffer, 0, 0, 0);
	
	stopTimer(timer);
	dTime = (cl_double)readTimer(timer);
	insertTimer("14.Release GL", dTime);

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
