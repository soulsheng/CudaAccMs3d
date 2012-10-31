set exe=updateVectorByMatrixPro

cd Release

if exist %exe%.txt del %exe%.txt

start /wait %exe% --help

notepad %exe%