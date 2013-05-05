#pragma once

#include "CL\cl.h"
#include "Vertex.h"
#include "Joint.h"

class CMatrixMulVector
{
public:
	CMatrixMulVector();
	~CMatrixMulVector();

public:
	// 数据初始化：坐标、矩阵
	void initialize(int sizeVertex, int sizeJoint);
	
	// 数据销毁：坐标、矩阵
	void unInitialize();

	// 执行各种算法
	void ExecuteNativeCPP();
	void ExecuteNativeSSE();
	
	// 验证结果是否正确
	bool verifyEqual();

private:
	// 矩阵变换
	void MatrixVectorMul(float* vIn, float* vOut, float* mat);
	void MatrixVectorMul(cl_float4* vIn, cl_float4* vOut, cl_float4* mat);

	// 验证结果是否正确
	bool verifyEqual(float *v, float* vRef, int size);
	bool verifyEqual(cl_float4 *v, cl_float4* vRef, int size);

public:
	// 数据定义
	Vertexes  _vertexesStatic;//静态顶点坐标
	Vertexes  _vertexesDynamic, _vertexesDynamicRef;//动态顶点坐标
	Joints		_joints;//关节矩阵
};