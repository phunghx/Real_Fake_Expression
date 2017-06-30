luapath=$1
cd libs/dlib-19.4/examples
g++ -std=c++11 -I.. ../dlib/all/source.cpp -lpthread -lX11 face_landmark_detection_lua.cpp -DDLIB_JPEG_SUPPORT /usr/lib/x86_64-linux-gnu/libjpeg.so -shared -fPIC -o liblandmark_detector.so -I$luapath/include -L$luapath/lib -L$luapath/lib/lua/5.1 -lTH -lluajit -lluaT 
cp liblandmark_detector.so ../../../training/liblandmark_detector.so
cp liblandmark_detector.so ../../../preprocessing/liblandmark_detector.so
cp liblandmark_detector.so ../../../liblandmark_detector.so
cd ../../..

