ffi = require("ffi")
-- Load myLib
myLib = ffi.load(paths.cwd() .. '/liblandmark_detector.so')
-- Function prototypes definition
ffi.cdef [[
   int faceLandmark(float* keyPoints, const char* folder,const char* shape_detector, bool training,bool vis, int dwx, int dwy);
]]

a = torch.FloatTensor(252,1,68*2):fill(0)
myLib.faceLandmark(torch.data(a), ffi.string("/dataset/Train_v1/40/D2N2Sur/*.jpg"),ffi.string("/data5/Real_Fake_Emotion/Real_Fake_Expression/libs/dlib-19.4/shape_predictor_68_face_landmarks.dat"),true,true,20,10)

npy4th = require 'npy4th'
npy4th.savenpy(string.format('./tmp/test.npy'),a)

--for i=1,252 do
--	npy4th.savenpy(string.format('./tmp/%04d.npy',i),a[i][1])
--end
--local dbg = require("debugger")
--dbg()
