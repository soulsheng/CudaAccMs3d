rem time = 0.53

set exe=updateVectorByMatrixPro

cd Release

if exist %exe%.txt del %exe%.txt

start /wait %exe% --problem=3 --quiet=0 --aligned=1 --separate=0 --sort=1 --memory=0

notepad %exe%