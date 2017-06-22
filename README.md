#This is the source code for Chalearn LAP Real Versus Fake Expressed Emotion Challenge @ICCV 2017 </br>
Author: Huynh Xuan Phung </br>
Email: phunghx <at> gmail <dot> com </br>
* Dependency libraries:
1. Torch 7: http://torch.ch/docs/getting-started.html
2. Python 2.4
3. OpenCV 2.4.9
4. dlib for python
5. Xgboost https://github.com/dmlc/xgboost
6. cunn library for torch

* Instructions:
1. Copy videos into folder data/test
2. run the command file: ./run.sh
4. final result is valid_prediction.pkl

Notes:
- Please detect face region manually on the first frame if the face detection tool can not detect the face. It has pop up window; left mouse click on the face region then right click to confirm.
