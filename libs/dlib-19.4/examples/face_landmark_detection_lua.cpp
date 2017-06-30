// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    


    This face detector is made using the classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  The pose estimator was created by
    using dlib's implementation of the paper:
        One Millisecond Face Alignment with an Ensemble of Regression Trees by
        Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset.  

    Also, note that you can train your own models using dlib's machine learning
    tools.  See train_shape_predictor_ex.cpp to see an example.

    


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/
#include <glob.h>
#include <luaT.h>
#include <TH/TH.h>
#include <TH/THStorage.h>
#include <TH/THTensor.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
//#define DLIB_JPEG_SUPPORT
//using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

extern "C"
{
	int faceLandmark(float* keyPoints, const char* folder,const char* shape_detector, bool training,bool vis, int dwx, int dwy);
}

int faceLandmark(float* keyPoints, const char* folder,const char* shape_detector, bool training,bool vis, int dwx, int dwy)
{  
    dlib::shape_predictor sp;
    dlib::deserialize(shape_detector) >> sp;
   
    glob_t glob_result;
    glob(folder,GLOB_TILDE,NULL,&glob_result);
    //keyPoints = (float*)malloc(glob_result.gl_pathc * 68*2*sizeof(float));
    float *temp_Keypoint;
    temp_Keypoint = keyPoints;
    int point_size = 68*2;
    //dlib::image_window win;
    
    
    int dx1,dy1,dx2,dy2;
    dx1=0;dy1=0;dx2=0;dy2=0;
    time_t t;
    if (training)	{
	srand((unsigned) time(&t));
	dx1 = rand() % dwx;dx2 = rand() % dwx;
	dy1 = rand() % dwy;dy2 = rand() % dwy;
    }
    for(unsigned int i=0; i<glob_result.gl_pathc; ++i){
  	//cout << glob_result.gl_pathv[i] << " "<<endl;
	dlib::array2d<dlib::rgb_pixel> img;
        dlib::load_image(img, glob_result.gl_pathv[i]);
	dlib::rectangle det = dlib::rectangle(0+dx1,0+dy1,img.nc()-dx2,img.nr()-dy2);
	dlib::full_object_detection shape = sp(img, det);
	/*
	if (vis)	{
	
		std::vector<dlib::full_object_detection> shapes;
        	std::vector<dlib::rectangle> dets;
		dets.push_back(det);
		shapes.push_back(shape);
        	win.clear_overlay();
        	win.set_image(img);
		win.add_overlay(dets, dlib::rgb_pixel(255,0,0));
        	win.add_overlay(render_face_detections(shapes));
	}
	*/
        for (unsigned long j = 0; j < shape.num_parts(); ++j)
        {
            keyPoints[i*point_size + j] = (float)1.0* shape.part(j)(0)/(img.nc()-dx1-dx2) ;
        }
        for (unsigned long j = 0; j < shape.num_parts(); ++j)
        {
            keyPoints[i*point_size + j+68] = (float)1.0*shape.part(j)(1)/(img.nr()-dy1-dy2) ;
        }	
    }
    return 1;
}

