set exe=updateVectorByMatrixPro

cd Debug

if exist %exe%.txt del %exe%.txt

for /l %%i in (0,1,6) do start /wait %exe% --problem=%%i

notepad %exe%