#This is the source code for Chalearn LAP Real Versus Fake Expressed Emotion Challenge @ICCV 2017 </br>
Author: Huynh Xuan Phung </br>
Email: phunghx 	&lt;at&gt; gmail &lt;dot&gt; com </br>
</br>
Title: Mirror neuron for Real versus Fake emotion recognition based on the movement of facial landmarks </br>
Author: Xuan-Phung Huynh </br>
HCI Lab, Sejong University, Korea </br>

* Training data: please download the preprocessing data from the link https://sejonguniversity-my.sharepoint.com/personal/phunghx_sju_ac_kr/_layouts/15/guestaccess.aspx?folderid=0b6feb31128fa428b82c54fc412f64d1a&authkey=AaVIHwtz95X0uCG_wve7Qz8

* Dependency libraries: 
0. Ubuntu 16.04
1. Torch 7: http://torch.ch/docs/getting-started.html
2. Python 2.4
3. OpenCV 2.4.9
4. dlib for python
5. Xgboost https://github.com/dmlc/xgboost
6. cunn library for torch
7. npy4th https://github.com/htwaijry/npy4th
8. Cuda 8.0
</br>

* Instructions:

1. Clone this repository into your local machine: https://github.com/phunghx/Real_Fake_Expression. I set this reposity on your machine is REALFAKE=/Real_Fake_Expression

If you use our pre-trained models please follow these steps:
1. Extract face regions from video
	- copy all video into a folder, we set it at $REALFAKE/data/test. The name of video is followed the challenge dataset: &lt;id&gt;_&lt;facial&gt;.mp4. &lt;facial&gt; is one of ANGER,CONTENTMENT,DISGUST,HAPPINESS,SADNESS,SURPRISE.

1. Copy testing videos into folder data/test
2. run the command file: ./run.sh
4. final result is valid_prediction.pkl

</br>
Notes:
- Please detect face region manually on the first frame if the face detection tool can not detect the face. It has pop up window; left mouse click on the face region then right click to confirm.

=======================================================
</br>
Training
1. Download training data from link https://drive.google.com/open?id=0B9dYHyzro_Q_d1hUamJGNndxSGM; extract data to a folder, for example /tmp/data/RealFake_trainset
2. run training via command : ./training.sh /tmp/data/RealFake_trainset
Note: training data is facial landmark point of training videos
</br>
Our paper in progression







