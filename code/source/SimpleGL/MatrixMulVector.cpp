
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

void CMatrixMulVector::ExecuteNativeCPP()
{
#if  VBO_MAP
	glBindBuffer( GL_ARRAY_BUFFER, vertexObj[0] );
	cl_float4* pVertex = (cl_float4*)glMapBuffer( GL_ARRAY_BUFFER, GL_READ_WRITE );
#endif


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
		cl_float4  blendpos, lastpos;

		for(int j= 0 ; j< SIZE_PER_BONE; j++){

			for (int k=0;k<3;k++)
			{
				blendpos.s[k]= 0.0f;
				lastpos.s[k]= 0.0f;
			}

			cl_float4 indexes = _vertexesStatic.pIndex[i];
			cl_float4 *pMat =  _joints.pMatrix + (int)indexes.s[j]*MATRIX_SIZE_LINE;

			MatrixVectorMul( _vertexesStatic.pVertex+i, &blendpos, pMat);

			cl_float4  weight = _vertexesStatic.pWeight[i];
			for (int k=0;k<3;k++)
			{
				lastpos.s[k]+= blendpos.s[k] * weight.s[j];
			}
		}

		for (int k=0;k<3;k++)
		{
#if  VBO_MAP
			pVertex[i].s[k] = lastpos.s[k] ;
#else
			_vertexesDynamicRef.pVertex[i].s[k] = lastpos.s[k] ;
#endif
		}
#endif
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
// 	if( g_pfInputBuffer ) {clReleaseMemObject( g_pfInputBuffer ); g_pfInputBuffer = NULL;}
// 	if( g_pfOCLOutputBuffer ) {clReleaseMemObject( g_pfOCLOutputBuffer ); g_pfOCLOutputBuffer = NULL;}
// 	if( g_pfOCLIndex ) {clReleaseMemObject( g_pfOCLIndex ); g_pfOCLIndex = NULL;}
// 	if( g_pfOCLMatrix ) {clReleaseMemObject( g_pfOCLMatrix ); g_pfOCLMatrix = NULL;}

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

void CMatrixMulVector::setupVBO(cl_context	pContext, cl_device_id pDevice_ID, cl_kernel pKernel, cl_command_queue pCmdQueue
																, int* nLocationAttrib )
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
	glBufferData(GL_ARRAY_BUFFER, _vertexesStatic.nSize * sizeof(cl_float4), (GLvoid *)_vertexesStatic.pIndex, GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray( _locationAttrib[0] );
	glVertexAttribPointer( _locationAttrib[0], 4, GL_FLOAT, GL_FALSE, 0, NULL);

	// 	Create Vertex buffer object: 顶点属性 矩阵权重
	glBindBuffer(GL_ARRAY_BUFFER, vertexObj[2]);
	glBufferData(GL_ARRAY_BUFFER, _vertexesStatic.nSize * sizeof(cl_float4), (GLvoid *)_vertexesStatic.pWeight, GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray( _locationAttrib[1] );
	glVertexAttribPointer( _locationAttrib[1], 4, GL_FLOAT, GL_FALSE, 0, NULL);

	glBindBuffer( GL_ARRAY_BUFFER, NULL );
	glBindVertexArray ( NULL ); 
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