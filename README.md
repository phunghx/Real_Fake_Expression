#This is the source code for Chalearn LAP Real Versus Fake Expressed Emotion Challenge @ICCV 2017 </br>
Author: Huynh Xuan Phung </br>
Email: phunghx <at> gmail <dot> com </br>
</br>
Title: Mirror neuron for Real versus Fake emotion recognition </br>
Author: Xuan-Phung Huynh </br>
HCI Lab, Sejong University, Korea </br>
* Dependency libraries: 
1. Torch 7: http://torch.ch/docs/getting-started.html
2. Python 2.4
3. OpenCV 2.4.9
4. dlib for python
5. Xgboost https://github.com/dmlc/xgboost
6. cunn library for torch
</br>
* Instructions:
1. Copy videos into folder data/test
2. run the command file: ./run.sh
4. final result is valid_prediction.pkl
</br>
Notes:
- Please detect face region manually on the first frame if the face detection tool can not detect the face. It has pop up window; left mouse click on the face region then right click to confirm.

=======================================================
</br>
Training
1. Download training data from link https://drive.google.com/open?id=0B9dYHyzro_Q_d1hUamJGNndxSGM; extract data to a folder 
2. run training via command : ./training.sh /<train dataset folder/>








