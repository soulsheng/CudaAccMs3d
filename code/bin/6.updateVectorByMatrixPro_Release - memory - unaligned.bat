set exe=updateVectorByMatrixPro

cd Release

if exist %exe%.txt del %exe%.txt

for /l %%i in (0,1,2) do (
for /l %%j in (0,1,2) do (
for /l %%m in (0,1,2) do (
start /wait /min %exe% --problem=%%k --quiet=1 --aligned=0 --separate=%%i --sort=%%j --memory=%%m
echo  aligned=0 >> %exe%.txt
echo  separate=%%i >> %exe%.txt
echo  sort=%%j >> %exe%.txt
echo  memory=%%m >> %exe%.txt
)
)
)

notepad %exe%