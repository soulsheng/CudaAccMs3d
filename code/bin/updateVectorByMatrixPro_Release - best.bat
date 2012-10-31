set exe=updateVectorByMatrixPro

cd Release

if exist %exe%.txt del %exe%.txt

start /wait %exe% --class=6 --aligned --mode=2 

notepad %exe%