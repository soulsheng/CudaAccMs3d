set exe=updateVectorByMatrixPro

cd Release

if exist %exe%.txt del %exe%.txt

for /l %%i in (0,1,2) do (
for /l %%j in (0,1,6) do start /wait %exe% --problem=%%j --aligned=1 --separate=%%i )

notepad %exe%