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

#include "stdafx.h"
#include "CL\cl.h"
#include "utils.h"
#include "Vertex.h"
#include "Joint.h"
//for perf. counters
#include <Windows.h>
#include <omp.h>
//we want to use POSIX functions
#pragma warning( push )
#pragma warning( disable : 4996 )

// OpenCL specific
cl_context	g_context = NULL;
cl_command_queue g_cmd_queue = NULL;
cl_program	g_program = NULL;
cl_kernel	g_kernel = NULL;
cl_kernel	g_kernel4 = NULL;
cl_uint     g_min_align = 0;
cl_device_id g_device_ID =0;
cl_event g_perf_event = NULL;

// Host arrays
//cl_float* g_pfInput = NULL;
//cl_float* g_pfRegularOutput = NULL;
//cl_float* g_pfOCLOutput = NULL;
//int g_szTask = 16*1024*1024;


//	global and local sizes of global and local worksizes
size_t g_szGlobalWork = 32768;
size_t g_szLocalWorkX = 8;
size_t g_szLocalWorkY = 8;

//	cl_mem objects used as parameters for kernels
cl_mem g_pfInputBuffer = NULL;
cl_mem g_pfOCLOutputBuffer = NULL;
cl_mem g_pfOCLIndex = NULL;
cl_mem g_pfOCLMatrix = NULL;

bool g_bUseRelaxedMath = false;
bool g_bUseHostPtr = false;
bool g_bAutoGroupSize = false;
bool g_bEnableProfiling = false;
bool g_bWarming = false;
bool g_bGather4 = false;

bool g_bRunOnPG = false;

LARGE_INTEGER g_PerfFrequency;
LARGE_INTEGER g_PerformanceCountNDRangeStart;
LARGE_INTEGER g_PerformanceCountNDRangeStop;
LARGE_INTEGER g_PerformanceCountReadStart;
LARGE_INTEGER g_PerformanceCountReadStop;
cl_double g_NDRangeTime;

LARGE_INTEGER g_PerformanceCountReferenceStart;
LARGE_INTEGER g_PerformanceCountReferenceStop;

#define    MEGA_SIZE     (1<<20)  // Mega, or million
#define    JOINT_SIZE    100

float    PROBLEM_SCALE[] ={ 0.25f, 0.5f, 1, 2, 4, 8, 16, 32 }; // 问题规模档次，8档，250K至32M，2倍递增
int    PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[2] ;// 问题规模, 初始设为1M，即一百万
int iClass=2;

int g_szTask ;

bool USE_OPENMP = false;
bool USE_SSE = false;

// 数据定义
Vertexes  _vertexesStatic;//静态顶点坐标
Vertexes  _vertexesDynamic, _vertexesDynamicRef;//动态顶点坐标
Joints		_joints;//关节矩阵

// 数据初始化：坐标、矩阵
void initialize(int problem_size, int joint_size)
{
	_joints.initialize( JOINT_SIZE );
	_vertexesStatic.initialize( g_szTask, JOINT_SIZE );
	_vertexesDynamic.initialize( g_szTask, JOINT_SIZE );
	_vertexesDynamicRef.initialize( g_szTask, JOINT_SIZE );
}

// 数据销毁：坐标、矩阵
void unInitialize()
{
	_joints.unInitialize();
	_vertexesStatic.unInitialize();
	_vertexesDynamic.unInitialize();
}
void Cleanup()
{
    //release g_kernel, g_program, and memory objects
    if( g_pfInputBuffer ) {clReleaseMemObject( g_pfInputBuffer ); g_pfInputBuffer = NULL;}
    if( g_pfOCLOutputBuffer ) {clReleaseMemObject( g_pfOCLOutputBuffer ); g_pfOCLOutputBuffer = NULL;}
	if( g_pfOCLIndex ) {clReleaseMemObject( g_pfOCLIndex ); g_pfOCLIndex = NULL;}
	if( g_pfOCLMatrix ) {clReleaseMemObject( g_pfOCLMatrix ); g_pfOCLMatrix = NULL;}
    if( g_kernel ) {clReleaseKernel( g_kernel );  g_kernel = NULL;}
    if( g_kernel4 ) {clReleaseKernel( g_kernel4 );  g_kernel4 = NULL;}
    if( g_program ) {clReleaseProgram( g_program );  g_program = NULL;}
    if( g_cmd_queue ) {clReleaseCommandQueue( g_cmd_queue );  g_cmd_queue = NULL;}
    if( g_context ) {clReleaseContext( g_context );  g_context = NULL;}
	if( g_perf_event ){clReleaseEvent(g_perf_event);g_perf_event =NULL;}
    //host memory
//    if(g_pfInput) {_aligned_free( g_pfInput ); g_pfInput = NULL;}
//    if(g_pfRegularOutput) {_aligned_free( g_pfRegularOutput ); g_pfRegularOutput = NULL;}
    //if(g_pfOCLOutput) {_aligned_free( g_pfOCLOutput ); g_pfOCLOutput = NULL;}
	unInitialize();
}

bool Setup_OpenCL( const char *program_source )
{
    cl_device_id devices[16];
    size_t cb;
    cl_uint size_ret = 0;
    cl_int err;
    int num_cores;
    char device_name[128] = {0};

	static const char buildOpts[] = "-cl-fast-relaxed-math";

	if(g_bRunOnPG)
	{
		printf("Trying to run on a Processor Graphics \n");
	}
	else
	{
		printf("Trying to run on a CPU \n");
	}

	cl_platform_id intel_platform_id = GetIntelOCLPlatform();
    if( intel_platform_id == NULL )
    {
        printf("ERROR: Failed to find Intel OpenCL platform.\n");
        return false;
    }

    cl_context_properties context_properties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)intel_platform_id, NULL };

    // create the OpenCL context on a CPU/PG 
	if(g_bRunOnPG)
	{
		g_context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
	}
	else
	{
		g_context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_CPU, NULL, NULL, NULL);
	}
    if (g_context == (cl_context)0)
        return false;

    // get the list of CPU devices associated with context
    err = clGetContextInfo(g_context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    clGetContextInfo(g_context, CL_CONTEXT_DEVICES, cb, devices, NULL);
	g_cmd_queue = clCreateCommandQueue(g_context, devices[0], g_bEnableProfiling ? CL_QUEUE_PROFILING_ENABLE: 0, NULL);
    if (g_cmd_queue == (cl_command_queue)0)
    {
        Cleanup();
        return false;
    }

    char *sources = ReadSources(program_source);	//read program .cl source file
    g_program = clCreateProgramWithSource(g_context, 1, (const char**)&sources, NULL, NULL);
    if (g_program == (cl_program)0)
    {
        printf("ERROR: Failed to create Program with source...\n");
        Cleanup();
        free(sources);
        return false;
    }

	err = clBuildProgram(g_program, 0, NULL, g_bUseRelaxedMath ? buildOpts :NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("ERROR: Failed to build program...\n");
        BuildFailLog(g_program, devices[0]);
        Cleanup();
        free(sources);
        return false;
    }
#if !VECTOR_FLOAT4
    g_kernel = clCreateKernel(g_program, "updateVectorByMatrix", NULL);
#else
	g_kernel = clCreateKernel(g_program, "updateVectorByMatrix4", NULL);
#endif
    if (g_kernel == (cl_kernel)0)
    {
        printf("ERROR: Failed to create kernel...\n");
        Cleanup();
        free(sources);
        return false;
    }
    g_kernel4 = clCreateKernel(g_program, "SimpleKernel4", NULL);
    if (g_kernel4 == (cl_kernel)0)
    {
        printf("ERROR: Failed to create second kernel...\n");
        Cleanup();
        free(sources);
        return false;
    }
    free(sources);

    // use first device ID
    g_device_ID = devices[0];
    err = clGetDeviceInfo(g_device_ID, CL_DEVICE_NAME, 128, device_name, NULL);
    if (err!=CL_SUCCESS)
    {
        printf("ERROR: Failed to get device information (device name)...\n");
        Cleanup();
        return false;
    }
    printf("Using device %s...\n", device_name);

    err = clGetDeviceInfo(g_device_ID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_cores, NULL);
    if (err!=CL_SUCCESS)
    {
        printf("ERROR: Failed to get device information (max compute units)...\n");
        Cleanup();
        return false;
    }
    printf("Using %d compute units...\n", num_cores);


    err = clGetDeviceInfo(g_device_ID, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &g_min_align, NULL);
    if (err!=CL_SUCCESS)
    {
        printf("ERROR: Failed to get device information (max memory base address align size)...\n");
        Cleanup();
        return false;
    }
    g_min_align /= 8; //in bytes
    printf("Buffer alignment required for zero-copying is %d bytes (CL_DEVICE_MEM_BASE_ADDR_ALIGN)\n\n", g_min_align);

    return true; // success...
}
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
bool verifyEqual(cl_float4 *v, cl_float4* vRef, int size)
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
void MatrixVectorMul(float* vIn, float* vOut, float* mat)
{
	vOut[0] = vIn[0] * mat[0] + vIn[1] * mat[1] + vIn[2] * mat[2]  + mat[3];
	vOut[1] = vIn[0] * mat[4] + vIn[1] * mat[5] + vIn[2] * mat[6]  + mat[7];
	vOut[2] = vIn[0] * mat[8] + vIn[1] * mat[9] + vIn[2] * mat[10]  + mat[11];
}
void MatrixVectorMul(cl_float4* vIn, cl_float4* vOut, cl_float4* mat)
{
	vOut->s[0] = vIn->s[0] * mat[0].s[0] + vIn->s[1] * mat[0].s[1] + vIn->s[2] * mat[0].s[2]  + mat[0].s[3];
	vOut->s[1] = vIn->s[0] * mat[1].s[0] + vIn->s[1] * mat[1].s[1] + vIn->s[2] * mat[1].s[2]  + mat[1].s[3];
	vOut->s[2] = vIn->s[0] * mat[2].s[0] + vIn->s[1] * mat[2].s[1] + vIn->s[2] * mat[2].s[2]  + mat[2].s[3];
}

void ExecuteNativeCPP()
{
	QueryPerformanceCounter(&g_PerformanceCountReferenceStart);

#if 1//use_openmp
#pragma omp parallel for
#endif
	for(int i=0;i<PROBLEM_SIZE;i++){

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

	QueryPerformanceCounter(&g_PerformanceCountReferenceStop);

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

void ExecuteNativeSSE()
{
	QueryPerformanceCounter(&g_PerformanceCountReferenceStart);

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
	QueryPerformanceCounter(&g_PerformanceCountReferenceStop);

}

bool ExecuteKernel()
{
    cl_int err = CL_SUCCESS;

    const cl_mem_flags INFlags  = (g_bUseHostPtr ? CL_MEM_USE_HOST_PTR: CL_MEM_COPY_HOST_PTR) | CL_MEM_READ_ONLY; 
    const cl_mem_flags OUTFlags = (g_bUseHostPtr ? CL_MEM_USE_HOST_PTR: CL_MEM_COPY_HOST_PTR) | CL_MEM_READ_WRITE;
	cl_int errcode_ret;
    // allocate buffers
#if !VECTOR_FLOAT4
    g_pfInputBuffer = clCreateBuffer(g_context, INFlags, g_szTask*VERTEX_VECTOR_SIZE * sizeof(cl_float) , _vertexesStatic.pVertex, &errcode_ret);
#else
	g_pfInputBuffer = clCreateBuffer(g_context, INFlags, g_szTask * sizeof(cl_float4) , _vertexesStatic.pVertex, &errcode_ret);
#endif
    if ( errcode_ret!=CL_SUCCESS )
    {
		printf("ERROR: Failed to create g_pfInputBuffer...\n");
    }
	if (g_pfInputBuffer == (cl_mem)0)
    {
        printf("ERROR: Failed to create g_pfInputBuffer...\n");
        return false;
    }
	g_pfOCLIndex = clCreateBuffer(g_context, INFlags, sizeof(cl_int)*g_szTask , _vertexesStatic.pIndex, NULL);
#if !VECTOR_FLOAT4
	g_pfOCLMatrix = clCreateBuffer(g_context, INFlags, sizeof(cl_float)*VERTEX_VECTOR_SIZE * MATRIX_SIZE_LINE*JOINT_SIZE , _joints.pMatrix, NULL);
#else
	g_pfOCLMatrix = clCreateBuffer(g_context, INFlags, sizeof(cl_float4) * MATRIX_SIZE_LINE*JOINT_SIZE , _joints.pMatrix, NULL);
#endif

#if !VECTOR_FLOAT4
	g_pfOCLOutputBuffer = clCreateBuffer(g_context, OUTFlags, sizeof(cl_float)*VERTEX_VECTOR_SIZE * g_szTask , _vertexesDynamic.pVertex, NULL);
#else
	g_pfOCLOutputBuffer = clCreateBuffer(g_context, OUTFlags, sizeof(cl_float4) * g_szTask , _vertexesDynamic.pVertex, NULL);
#endif    
	if (g_pfOCLOutputBuffer == (cl_mem)0)
    {
        printf("ERROR: Failed to create g_pfOCLOutputBuffer...\n");
        return false;
    }

	cl_kernel	kernel = g_kernel;
    size_t globalWorkSize[2];
    size_t localWorkSize[2];
    globalWorkSize[0] = (size_t)sqrtf(g_szGlobalWork);
	globalWorkSize[1] = globalWorkSize[0];
	if(g_bGather4)
	{
	    globalWorkSize[0]/=4; //since proccesing in quadruples
		kernel = g_kernel4;
	}

    //Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &g_pfInputBuffer);
    if (err != CL_SUCCESS)
    {
        printf("ERROR: Failed to set input g_kernel arguments...\n");
        return false;
    }
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &g_pfOCLIndex);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &g_pfOCLMatrix);

    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &g_pfOCLOutputBuffer);
    if (err != CL_SUCCESS)
    {
        printf("ERROR: Failed to set input g_kernel arguments...\n");
        return false;
    }

    localWorkSize[0] = g_szLocalWorkX;
	localWorkSize[1] = g_szLocalWorkY;
    printf("Original global work size (%lu, %lu)\n", globalWorkSize[0], globalWorkSize[1]);
    printf("Original local work size (%lu, %lu)\n", localWorkSize[0], localWorkSize[1]);
	if(g_bAutoGroupSize)
	{
		printf("Run-time determines optimal workgroup size\n\n");
	}
	size_t  workGroupSizeMaximum;
	err = clGetKernelWorkGroupInfo(g_kernel, g_device_ID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void *)&workGroupSizeMaximum, NULL);
	printf("Maximum workgroup size for this kernel  %lu\n\n",workGroupSizeMaximum );

	if ( g_szGlobalWork>workGroupSizeMaximum )
	{
		globalWorkSize[0] = workGroupSizeMaximum;
		globalWorkSize[1] = g_szGlobalWork / workGroupSizeMaximum;
	}
	printf("Actual global work size (%lu, %lu)\n", globalWorkSize[0], globalWorkSize[1]);

	if(g_bWarming)
	{
		printf("Warming up OpenCL execution...");
		err= clEnqueueNDRangeKernel(g_cmd_queue, kernel, 2, NULL, globalWorkSize, g_bAutoGroupSize? NULL:localWorkSize, 0, NULL, NULL);
		clFinish(g_cmd_queue);
		printf("Done\n");
	}

    printf("Executing OpenCL kernel...");
    QueryPerformanceCounter(&g_PerformanceCountNDRangeStart);
	// execute kernel, pls notice g_bAutoGroupSize
	err= clEnqueueNDRangeKernel(g_cmd_queue, kernel, 2, NULL, globalWorkSize, g_bAutoGroupSize? NULL:localWorkSize, 0, NULL, &g_perf_event);
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
    QueryPerformanceCounter(&g_PerformanceCountNDRangeStop);
    printf("Done\n");

    if(g_bEnableProfiling)
    {    
        cl_ulong start = 0;
        cl_ulong end = 0;

        //notice that pure HW execution time is END-START
		err = clGetEventProfilingInfo(g_perf_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err != CL_SUCCESS)
        {
            printf("ERROR: Failed to get clGetEventProfilingInfo CL_PROFILING_COMMAND_START...\n");
            return false;
        }
        err = clGetEventProfilingInfo(g_perf_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        if (err != CL_SUCCESS)
        {
            printf("ERROR: Failed to get clGetEventProfilingInfo CL_PROFILING_COMMAND_END...\n");
            return false;
        }
        g_NDRangeTime = (cl_double)(end - start)*(cl_double)(1e-06);
    }

    void* tmp_ptr = NULL;
    QueryPerformanceCounter(&g_PerformanceCountReadStart);
	if(g_bUseHostPtr)
    {
#if !VECTOR_FLOAT4
        tmp_ptr = clEnqueueMapBuffer(g_cmd_queue, g_pfOCLOutputBuffer, true, CL_MAP_READ, 0, sizeof(cl_float)*VERTEX_VECTOR_SIZE * g_szTask , 0, NULL, NULL, NULL);
#else
		tmp_ptr = clEnqueueMapBuffer(g_cmd_queue, g_pfOCLOutputBuffer, true, CL_MAP_READ, 0, sizeof(cl_float4) * g_szTask , 0, NULL, NULL, NULL);
#endif
        if(tmp_ptr!=_vertexesDynamic.pVertex)
        {
            printf("ERROR: clEnqueueMapBuffer failed to return original pointer\n");
            return false;
        }
    }
    else
    {
#if !VECTOR_FLOAT4
		err = clEnqueueReadBuffer(g_cmd_queue, g_pfOCLOutputBuffer, CL_TRUE, 0, sizeof(cl_float) *VERTEX_VECTOR_SIZE* g_szTask , _vertexesDynamic.pVertex, 0, NULL, NULL);
#else
		err = clEnqueueReadBuffer(g_cmd_queue, g_pfOCLOutputBuffer, CL_TRUE, 0, sizeof(cl_float4) * g_szTask , _vertexesDynamic.pVertex, 0, NULL, NULL);
#endif
		if (err != CL_SUCCESS)
        {
            printf("ERROR: Failed to clEnqueueReadBuffer...\n");
            return false;
        }
    }
    clFinish(g_cmd_queue);
	QueryPerformanceCounter(&g_PerformanceCountReadStop);

    clEnqueueUnmapMemObject(g_cmd_queue, g_pfOCLOutputBuffer, tmp_ptr, 0, NULL, NULL);
	clReleaseMemObject(g_pfInputBuffer); g_pfInputBuffer = NULL;
    clReleaseMemObject(g_pfOCLOutputBuffer); g_pfOCLOutputBuffer = NULL;
	clReleaseMemObject(g_pfOCLIndex); g_pfOCLIndex = NULL;
	clReleaseMemObject(g_pfOCLMatrix); g_pfOCLMatrix = NULL;
    return true;
}

static cl_float randomFloat(const cl_float fMinValue, const cl_float fMaxValue)
{
    if (fMinValue == fMaxValue)
    {
        return fMaxValue;
    }
    cl_float tmp = (cl_float)(rand() % (int)(fMaxValue - fMinValue));
    tmp = (0 == tmp) ? 1.0f : tmp;
    return fMinValue + ((cl_float)(rand() % (int)(fMaxValue - fMinValue))) / tmp;
}

void Usage()
{
    printf("Usage: SimpleOptimizations.exe [--h] [-t <TaskSize>][-l <GroupSize>] [-r] [-p] [-a] [-w] [-v] [-g]\n");
    printf("  where, --h prints this message\n");
    printf("    <TaskSize> is task size (also global size)\n");
    printf("    <GroupSize> is work group size (aka local size)\n");
    printf("    -r relaxed math enabled\n");
    printf("    -p host pointers/buffer-mapping enabled\n");
    printf("    -a auto-selected work group size enbaled, [-l] 'local size' option will be ignored in this case\n");
    printf("    -f OpenCL profiling will be enabled\n");
    printf("    -w additional \"warming\" kernel run enabled \n");
    printf("    -v \"gather4\" kernel version\n");
    printf("    -g run on Processor Graphics\n");
    exit(-1);
}

// main execution routine - perform simple math on float vectors
int _tmain(int argc, _TCHAR* argv[])
{
    //parse command line
    int argn = 1;
    while (argn < argc)
    {
        if (_tcscmp(argv[argn], _T("--h")) == 0)
        {
            Usage();
        }
        else if (_tcscmp(argv[argn], _T("-t")) == 0)
        {
            if(++argn==argc)
                Usage();
            g_szTask = _ttoi(argv[argn]);
            argn++;
        }
        else if (_tcscmp(argv[argn], _T("-l")) == 0)
        {
            if(++argn==argc)
                Usage();
            g_szLocalWorkX = _ttoi(argv[argn]);
            argn++;
        }
        else if (_tcscmp(argv[argn], _T("-w")) == 0)
        {
            g_bWarming= true;
            argn++;
        }
        else if (_tcscmp(argv[argn], _T("-v")) == 0)
        {
            g_bGather4 = true;
            argn++;
        }
		else if (_tcscmp(argv[argn], _T("-r")) == 0)
        {
            g_bUseRelaxedMath = true;
            argn++;
        }
        else if (_tcscmp(argv[argn], _T("-p")) == 0)
        {
            g_bUseHostPtr = true;
            argn++;
        }
        else if (_tcscmp(argv[argn], _T("-a")) == 0)
        {
            g_bAutoGroupSize = true;
            argn++;
        }
        else if (_tcscmp(argv[argn], _T("-f")) == 0)
        {
            g_bEnableProfiling = true;
            argn++;
        }
		else if (_tcscmp(argv[argn], _T("-g")) == 0)
		{
			g_bRunOnPG = true;
			argn++;
		}
		else if (_tcscmp(argv[argn], _T("-index")) == 0)
		{
			if(++argn==argc)
				Usage();
			iClass = _ttoi(argv[argn]);
			argn++;
		}
        else
        {
            argn++;
        }
    }
    if( argc < 2 )
    {
        printf("No command line arguments specified, using default values.\n");
    }

    //initialize Open CL objects (context, queue, etc.)
    if( Setup_OpenCL("SimpleOptimizations.cl")!=true )
        return -1;
	
	// 问题规模档次，7档，64K至256M，4倍递增
	PROBLEM_SIZE  = MEGA_SIZE * PROBLEM_SCALE[iClass] ;

	g_szTask = PROBLEM_SIZE;
    g_szGlobalWork = g_szTask;
	if(g_szGlobalWork%(g_szLocalWorkX * g_szLocalWorkX) !=0 && !g_bAutoGroupSize)
	{
		printf("Global or local work size is incorect.\n");
		printf("g_szGlobalWork / g_szLocalWork remainder is = %lu. This value must be 0\n", g_szGlobalWork%(g_szLocalWorkX * g_szLocalWorkX) );
        Cleanup();
        return -1;
    }

    //	set input array to random legal values
	srand(2011);

	// 数据初始化：坐标、矩阵
	initialize(PROBLEM_SIZE, JOINT_SIZE);

// 	for (int i = 0; i < g_szTask ; i += 1)
// 	{
// 		g_pfInput[i] = randomFloat(5.0f, 255.0f);
// 	}

    //retrieve perf. counter frequency
    QueryPerformanceFrequency(&g_PerfFrequency);

    //do simple math
    if(ExecuteKernel()!=true)
    {
        printf("Failed executing OpenCL kernel...\n");
        Cleanup();
        return -1;
    }

	printf("Executing reference...");
	if(g_bGather4)	
		ExecuteNativeSSE();
	else
		ExecuteNativeCPP();

	printf("Done\n\n");


    printf("NDRange perf. counter time %f ms.\n", 
        1000.0f*(float)(g_PerformanceCountNDRangeStop.QuadPart - g_PerformanceCountNDRangeStart.QuadPart)/(float)g_PerfFrequency.QuadPart);
    if(g_bEnableProfiling)    
        printf("NDRange event profiling time %f ms.\n\n", g_NDRangeTime);
    printf("Read buffer perf. counter time %f ms.\n", 
        1000.0f*(float)(g_PerformanceCountReadStop.QuadPart - g_PerformanceCountReadStart.QuadPart)/(float)g_PerfFrequency.QuadPart);

	printf("Reference. counter time %f ms.\n", 
		1000.0f*(float)(g_PerformanceCountReferenceStop.QuadPart - g_PerformanceCountReferenceStart.QuadPart)/(float)g_PerfFrequency.QuadPart);

    //Do verification
    printf("Performing verification...\n");
 	bool result = verifyEqual(_vertexesDynamic.pVertex, _vertexesDynamicRef.pVertex, PROBLEM_SIZE);
	printf("%s", !result ?"ERROR: Verification failed.\n":"Verification succeeded.\n");
    
    Cleanup();
    if(!result)
    {
		return -1;
	}
    else
    {
		return 0;
    }
}

#pragma warning( pop )
