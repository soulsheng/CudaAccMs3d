set exe=updateVectorByMatrixPro

cd Release

if exist %exe%.txt del %exe%.txt

for /l %%i in (0,1,2) do (
for /l %%j in (0,1,6) do (
start /wait /min %exe% --problem=%%j --quiet=1 --aligned=1 --separate=%%i 
)
echo  aligned=1 >> %exe%.txt
echo  separate=%%i >> %exe%.txt
)

notepad %exe%