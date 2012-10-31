set exe=updateVectorByMatrixPro

cd Release

if exist %exe%.txt del %exe%.txt

for /l %%j in (0,1,2) do (
for /l %%k in (0,1,6) do (
start /wait /min %exe% --problem=%%k --quiet=1 --aligned=1 --separate=0 --sort=%%j
)
echo  aligned=1 >> %exe%.txt
echo  separate=0 >> %exe%.txt
echo  sort=%%j >> %exe%.txt)
)

notepad %exe%