testshape=$1
testImage=$2
th test_data.lua -datasave dataTest -datainput $testshape	#data/shapeTest
python findFaces.py --image_folder $testImage --save testPairs.pkl  #preprocessing/data/testImage
python predict_test_66_7_percent.py --image_folder dataTest --save test_prediction.pkl
