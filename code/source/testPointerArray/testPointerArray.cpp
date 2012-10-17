// testPointerArray.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "PointerArray.h"

PointerArray	pa;
PointerArray	paRef;

void updatePointerArray(PointerArray& p, PointerArray& pRef)
{
	for (int i=0;i<SIZE_ARRAY;i++)
	{
		memcpy( p._pointerArray[i], pRef._pointerArray[i], sizeof(float)*SIZE_BUFFER );
	}

	for (int i=0;i<SIZE_ARRAY;i++)
	{
		for (int j=0;j<SIZE_BUFFER;j++)
		{
			p._pointerArray[i][j] += 10000.f;
		}
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	pa.initialize();
	paRef.initialize();

	updatePointerArray(pa, paRef);

	pa.unInitialize();
	paRef.unInitialize();

	return 0;
}

