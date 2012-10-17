#include "stdafx.h"
#include "PointerArray.h"

#include <stdlib.h>
#include <string.h>

PointerArray::PointerArray()
{
	for (int i=0;i<SIZE_ARRAY;i++)
	{
		_pointerArray[i] = NULL;
	}
}

PointerArray::~PointerArray()
{
	unInitialize();
}

void PointerArray::initialize()
{
	for (int i=0;i<SIZE_ARRAY;i++)
	{
		_pointerArray[i] = new float[SIZE_BUFFER];
		for (int j=0;j<SIZE_BUFFER;j++)
		{
			_pointerArray[i][j] = rand() * 1.0f;
		}
	}
}

void PointerArray::unInitialize()
{
	for (int i=0;i<SIZE_ARRAY;i++)
	{
		if(_pointerArray[i] != NULL)
		{
			delete[] _pointerArray[i];
			_pointerArray[i] = NULL;
		}
	}
}
