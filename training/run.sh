datastr=$1
echo  "$datastr"
th run.lua -facial anger -data $datastr &
th run.lua -facial contentment -data $datastr &
th run.lua -facial disgust -data $datastr &
th run.lua -facial happy -data $datastr &
th run.lua -facial sadness -data $datastr &
th run.lua -facial surprise  -data $datastr
