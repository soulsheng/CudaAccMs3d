set exe=updateVectorByMatrixPro

cd Release

if exist %exe%.txt del %exe%.txt

for /l %%i in (0,1,6) do start /wait %exe% --class=%%i

notepad %exe%