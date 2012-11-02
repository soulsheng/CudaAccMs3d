set exe=updateVectorByMatrixPro

cd Release

if exist %exe%.txt del %exe%.txt

for /l %%i in (0,1,2) do (
for /l %%j in (0,1,2) do (
for /l %%k in (0,1,6) do (
start /wait /min %exe% --problem=%%k --quiet=1 --aligned=1 --separate=2 --sort=%%j --memory=%%i
)
echo  aligned=1 >> %exe%.txt
echo  separate=2 >> %exe%.txt
echo  sort=%%j >> %exe%.txt
echo  memory=%%i >> %exe%.txt
)
)
)

notepad %exe%