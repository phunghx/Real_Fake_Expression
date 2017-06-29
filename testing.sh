th test_v2.lua -datasave dataTest -datainput data/shapeTest
python findFaces.py --image_folder preprocessing/data/testImage --save testPairs.pkl
python predict_test_66_7_percent.py
