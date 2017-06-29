local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')
--local options = opt
--local opClassTrain = classTrain
donkeys = Threads(
         3,
         function()
            require 'torch'
         end,
         function(idx)
            --opt = options
	    --classTrain = opClassTrain
            tid = idx
            local seed = 1 + idx
            torch.manualSeed(seed)
            print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
            ffi = require("ffi")
	    -- Load myLib
	    myLib = ffi.load(paths.cwd() .. '/liblandmark_detector.so')
	    -- Function prototypes definition
	    ffi.cdef [[
		   int faceLandmark(float* keyPoints, const char* folder,const char* shape_detector, bool training,bool vis, int dwx, int dwy);
	    ]]
         end
      );


local function sampleHookTrain(input)
   collectgarbage()
   local input_size = 68
   local output_size = 40
   local point = {18,19,20,21,22,23,24,25,26,27,31,32,33,34,35,36,37,38,39,40,41,
	42,43,44,45,46,47,48,49,51,52,53,55,57,58,59}
   local out = torch.FloatTensor(input:size(1),1,output_size*2)
   for i=1,#point do
	out[{{},1,i}] = input[{{},1,point[i]}]
	out[{{},1,i+output_size}] = input[{{},1,point[i]+input_size}]
   end
   local combine={{1,18},{27,17},{3,30},{15,30}}
   for i=1,#combine do
	out[{{},1,i+#point}] = (input[{{},1,combine[i][1]}] + input[{{},1,combine[i][2]}]) / 2
	out[{{},1,i+#point+output_size}] = (input[{{},1,combine[i][1]+input_size}] + input[{{},1,combine[i][2]+input_size}]) / 2
   end
   out:mul(2):add(-1)
   return out
end
function makesurePath(pathData)
   if os.execute('[ -e ' .. pathData ..' ]') == nil then
   	os.execute('mkdir ' .. pathData )
   end
end
cmd = torch.CmdLine()
cmd:option('-input', '/data/Train', 'path of face region inputs')
cmd:option('-output', '/data/shape', 'path of saving landmark')
opt = cmd:parse(arg or {})

local dataPath = opt.input
local dataSave = out.output

--local dataPath = '../data/validImage/'
--local dataSave = './shapeValid/'

for dir in paths.iterdirs(dataPath) do
	makesurePath(dataSave .. dir)
end
for epoch=1,20 do
for dir in paths.iterdirs(dataPath) do
      donkeys:addjob(
        function()
	    local maxlength = 0
	    for file in paths.iterfiles(dataPath .. '/' .. dir) do
		maxlength = maxlength + 1
	    end
	    local data_full = torch.FloatTensor(maxlength,1,68*2)
	    myLib.faceLandmark(torch.data(data_full), ffi.string(dataPath .. dir .. '/*.jpg'),ffi.string("./shape_predictor_68_face_landmarks.dat"),true,false,20,10)
	    data_full = sampleHookTrain(data_full[{{2,data_full:size(1)}}])	
	    torch.save(dataSave .. dir .. '/'  .. string.format('/%04d.t7',epoch), data_full)
       end
     )
end
end
donkeys:synchronize()




