#This is the source code for Chalearn LAP Real Versus Fake Expressed Emotion Challenge @ICCV 2017 </br>
Author: Huynh Xuan Phung </br>
Email: phunghx <at> gmail <dot> com </br>

Instructions:
1. Copy videos into folder data/test
2. setup the latest caffe library and build for python: make pycaffe
2. change the path of caffe_path in preprocessing/_init_paths.py
3. run the command file: ./run.sh
4. final result is valid_prediction.pkl

Notes:
- Please detect face region manually on the first frame if the face detection tool can not detect the face. It has pop up window; using left click and drag to select face region then right click to confirm
