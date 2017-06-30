datastr=$1
cd preprocessing
th generateTraining.lua -input $datastr -output ../data/shapeTrain
cd ..
