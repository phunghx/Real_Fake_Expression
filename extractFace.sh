testdata=$1
cd preprocessing
python face_box_v2.py --video_folder $testdata --save testImage
th generateTesting.lua -input testImage -output ../data/shapeTest
cd ..


