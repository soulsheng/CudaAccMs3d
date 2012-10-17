#include "stdafx.h"
#include "PointerArray.h"

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
