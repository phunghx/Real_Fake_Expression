#This is the source code for Chalearn LAP Real Versus Fake Expressed Emotion Challenge @ICCV 2017 </br>
Author: Huynh Xuan Phung </br>
Email: phunghx 	&lt;at&gt; gmail &lt;dot&gt; com </br>
</br>
Title: A LSTM network with Parametric Bias and its Application to Real versus Fake Emotion Recognition </br>
Author: Xuan-Phung Huynh </br>
HCI Lab, Sejong University, Korea </br>


* Dependency libraries: 
0. Ubuntu 16.04
1. Torch 7: http://torch.ch/docs/getting-started.html
2. Python 2.7
3. OpenCV 2.4.9
4. Xgboost https://github.com/dmlc/xgboost
5. cunn library for torch
6. npy4th https://github.com/htwaijry/npy4th
7. Cuda 8.0
</br>

* Instructions:

1. Clone this repository into your local machine: https://github.com/phunghx/Real_Fake_Expression. I set this reposity on your machine is REALFAKE=/Real_Fake_Expression. 
2. Dataset: there are 4 ziped files as following:(link https://sejonguniversity-my.sharepoint.com/personal/phunghx_sju_ac_kr/_layouts/15/guestaccess.aspx?folderid=0b6feb31128fa428b82c54fc412f64d1a&authkey=AaVIHwtz95X0uCG_wve7Qz8 )
	- Train.zip: 40 subjects for training that include face region  only. Please download this file if you want to train the LSTMPB models
	- testImage.zip: face regions for testing video.
	- validImage.zip: face regions for validation video
	- shapeTrain.zip: face landmarks of training videos that we use for training xgboost models
	- shapeTest.zip: face landmarks of testing videos for our submission on challenge
	- shapeValid.zip: face landmarks of validating videos for submission on challenge

3. Build and install dependence libraris:
	- cuda 8.0
	- torch 7.1
	- python 2.7
	- opencv 2.4.9 for python
	- xgboost : sudo pip install xgboost
	- sklean: sudp pip install sklearn --upgrade
	- scikit-image: sudo pip install scikit-image --upgrade
	- libjpeg: sudo apt-get install libjpeg-dev
	- build dlib. From REALFAKE folder, run ./buildLibs.sh &lt;torch installed folder &gt;. For example, I install my torch at /libs/torch then I run ./buildLibs.sh /libs/torch/install

4. Replicate our results on the challenge website
	- delete all folders and files in REALFAKE/data
	- clear data ./cleardata.sh
	- download testImage.zip and extract to REALFAKE/data (REALFAKE/data/testImage)
	- download shapeTest.zip and extract to REALFAKE/data (REALFAKE/data/shapeTest)
	- run ./testing_data.sh REALFAKE/data/shapeTest REALFAKE/data/testImage
	- Final result is the file test_prediction.pkl in REALFAKE folder
	- If you want to see the result on the validation set, please download the validImage.zip and shapeValid.zip

5. Training
	- download and extract Train.zip into your machine. I assume the path of your extraction is /data/Train which contains 40 folders for 40 subjects.
	- Train LSTMPB : ./training.sh /data/Train
	- Generate landmark for training data: you can download shapeTrain.zip and extract to REALFAKE/data/shapeTrain (contain of 40 folders) or run ./createlandmarkData.sh /data/Train
	- Genetate PB vectors for training xgboost: ./createTrainData.sh REALFAKE/data/shapeTrain
	- Training xgboost: ./training_xgboost.sh

6. Testing

	- Copy all video into a folder, we set it at $REALFAKE/data/test. The name of video is followed the challenge dataset: &lt;id&gt;_&lt;facial&gt;.mp4. &lt;facial&gt; is one of ANGER,CONTENTMENT,DISGUST,HAPPINESS,SADNESS,SURPRISE.
	- extract face regions from testing video: ./extractFace.sh  $REALFAKE/data/test
	- testing: ./testing.sh REALFAKE/data/shapeTest REALFAKE/data/testImage
	- Final result is the file test_prediction.pkl in REALFAKE folder

</br>
Notes:
- Please detect face region manually on the first frame if the face detection tool can not detect the face. It has pop up window; left mouse click on the face region then right click to confirm.

=======================================================
</br>

Our paper in progression
</br>






