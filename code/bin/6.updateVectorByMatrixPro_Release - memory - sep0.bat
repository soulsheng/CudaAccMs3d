set exe=updateVectorByMatrixPro

cd Release

if exist %exe%.txt del %exe%.txt

for /l %%i in (0,1,2) do (
for /l %%j in (0,1,2) do (
start /wait /min %exe% --problem=3 --quiet=1 --aligned=1 --separate=0 --sort=%%i --memory=%%j
echo  aligned=1 >> %exe%.txt
echo  separate=0 >> %exe%.txt
echo  sort=%%i >> %exe%.txt
echo  memory=%%j >> %exe%.txt
)
)

notepad %exe%