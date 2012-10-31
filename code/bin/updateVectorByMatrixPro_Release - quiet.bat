set exe=updateVectorByMatrixPro

cd Release

if exist %exe%.txt del %exe%.txt

start /wait %exe% --quiet=1

notepad %exe%