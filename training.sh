datastr=$1
cd training
./run.sh $datastr
cd boostingTrain
./train_xgboost.sh

