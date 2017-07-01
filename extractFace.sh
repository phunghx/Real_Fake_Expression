testdata=$1
cd preprocessing
python face_box_v2.py --video_folder $testdata --save ../data/testImage
th generateTesting.lua -input ../data/testImage -output ../data/shapeTest
cd ..


