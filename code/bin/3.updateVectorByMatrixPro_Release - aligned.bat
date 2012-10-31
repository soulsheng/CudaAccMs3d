set exe=updateVectorByMatrixPro

cd Release

if exist %exe%.txt del %exe%.txt

for /l %%i in (0,1,6) do (
start /wait /min %exe% --problem=%%i --quiet=1 --aligned=1
)
echo  aligned=1 >> %exe%.txt

notepad %exe%