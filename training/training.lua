npy4th = require 'npy4th'
local optim = require 'optim'
local dbg = require("debugger")


local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')
local options = opt
local opClassTrain = classTrain
donkeys = Threads(
         2,
         function()
            require 'torch'
         end,
         function(idx)
            opt = options
	    classTrain = opClassTrain
            tid = idx
            local seed = 1 + idx
            torch.manualSeed(seed)
            print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
            ffi = require("ffi")
	    -- Load myLib
	    myLib = ffi.load(paths.cwd() .. '/liblandmark_detector.so')
	    -- Function prototypes definition
	    ffi.cdef [[
		   int faceLandmark(float* keyPoints, const char* folder,const char* shape_detector, bool training);
	    ]]
         end
      );

optimStateIn2H = {
         learningRate = 0.01,
         learningRateDecay = 0.0,
         momentum = 0.9,
         dampening = 0.0,
         weightDecay = 0
      }
local PB_learningrate = 0.9

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat('log', 'train_' .. opt.facial .. '.log'))
local batchNumber
local top1_epoch, loss_epoch

local prev_loss = 10000
local off_epoch = 0
local stop_epoch = 0

local function sampleHookTrain(input)
   collectgarbage()
   --local input, label = loadImage(path)
   --binary
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
   --local noise = torch.FloatTensor(1,2*output_size):normal(0,0.01)
   --if torch.uniform() > 0.5 then
	--   out[{{},1,{1,output_size}}]:add(x)
	--   out[{{},1,{output_size+1,2*output_size}}]:add(y)
	--for i=1,out:size(1) do
--		out[i]:add(noise)
--	end
 --  end

   out:mul(2):add(-1)
   return out

end
function clone_list(tensor_list, zero_too)
    -- utility function. todo: move away to some utils file?
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

local function sampleData(class,subject)
   local dataTable = {}
   local scalarTable = {}
   i = class
   local maxlength = 0
   for file in paths.iterfiles(opt.data .. '/' .. subject .. '/' .. classTrain[class]) do
	maxlength = maxlength + 1
   end
   local data_full = torch.FloatTensor(maxlength,1,68*2)
   --[[
   --if batchNumber==42 then dbg() end

   for j=2,maxlength do
	local shape = npy4th.loadnpy(opt.data .. '/' .. subject .. '/' .. classTrain[class] .. string.format('/%04d.npy',j))
	data_full[j-1][1]:copy(shape)
   end 
   --]]
   myLib.faceLandmark(torch.data(data_full), ffi.string(opt.data .. '/' .. subject .. '/' .. classTrain[class] .. '/*.jpg'),ffi.string("/data5/Real_Fake_Emotion/Real_Fake_Expression/libs/dlib-19.4/shape_predictor_68_face_landmarks.dat"),true)
   return sampleHookTrain(data_full[{{2,data_full:size(1)}}])
end
iterations = 80
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   batchNumber = 0
   cutorch.synchronize()
   model:training()
   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0
   loss_f_epoch = 0
   for i=1,iterations do
	donkeys:addjob(
	   function()
	    local class = math.max(1, math.ceil(torch.uniform() * 2))	
	    local subject = math.max(1, math.ceil(torch.uniform() * 40))
            local inputs = sampleData(class,subject)
	    return inputs,class
	   end,
           trainBatch
	)
   end
   donkeys:synchronize()
   cutorch.synchronize()
   loss_epoch = loss_epoch / iterations
   trainLogger:add{
      ['avg loss (train set) ferr'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.4f  \t ',
                       epoch, tm:time().real, loss_epoch))
   print('\n')
   -- save model
   collectgarbage()
   model:clearState()
   

   if loss_epoch < prev_loss then
	   torch.save(paths.concat('../' .. opt.facial, 'model.t7'), model)
           torch.save(paths.concat('../' .. opt.facial, 'PB_.t7'), PB:float())
	   prev_loss = loss_epoch
	   --[[
	   stop_epoch = 0
	   model:evaluate()
	   for i=1,40 do
		for class=1,2 do
			inputs = sampleData(class,i)
			validBatch(inputs,class,i)
		end
	   end 
	   -- ]]
   else	
	print("Model is not good") 	
	off_epoch = off_epoch + 1
	if off_epoch > 10 then 
		optimStateIn2H.learningRate = optimStateIn2H.learningRate * 0.9 
		PB_learningrate = PB_learningrate * 0.9
		off_epoch = 0	
		stop_epoch = stop_epoch + 1
	end
	
   end
   -----------------------------------
   


end
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local input_size = 40*2
local sizeBatch = 1
input = torch.CudaTensor(sizeBatch,input_size)
labels = torch.CudaTensor(sizeBatch,input_size)


local timer = torch.Timer()
local dataTimer = torch.Timer()

local threshold = 1
local grad_clip = 5

w, dW = model:getParameters()


local lstm_size = 512
init_state = {}
local h_init = torch.zeros(sizeBatch, lstm_size)
h_init = h_init:cuda()
for L=1, 4 do --num layer = 4, change in model also
   table.insert(init_state, h_init:clone())
   table.insert(init_state, h_init:clone())
end
function get_input_mem_cell()
    local input_mem_cell = torch.zeros(sizeBatch, lstm_size)
    input_mem_cell = input_mem_cell:float():cuda()
    return input_mem_cell
end


init_state_global = clone_list(init_state)

function computMatric(label, output, Error)
	local ferr = criterion:forward(label:squeeze(),output:squeeze())
	local f = 0
	f = Error:sum()
	return ferr, f
end
local window = 40
local k1 =0.8
local k2 = 0.5
init_state_model = clone_list(init_state)
for k, v in ipairs(model.forwardnodes) do
    if v.data.annotations.name == 'parametric_bias' then
	module_PB = v.data.module   
    elseif v.data.annotations.name == 'c_t_1' then 
	init_state_model[1] =  v.data.module   
    elseif v.data.annotations.name == 'h_t_1' then 
	init_state_model[2] =  v.data.module   
    elseif v.data.annotations.name == 'c_t_2' then 
	init_state_model[3] =  v.data.module   
    elseif v.data.annotations.name == 'h_t_2' then 
	init_state_model[4] =  v.data.module   
    elseif v.data.annotations.name == 'c_t_3' then 
	init_state_model[5] =  v.data.module   
    elseif v.data.annotations.name == 'h_t_3' then 
	init_state_model[6] =  v.data.module   
    elseif v.data.annotations.name == 'c_t_4' then 
	init_state_model[7] =  v.data.module   
    elseif v.data.annotations.name == 'h_t_4' then 
	init_state_model[8] =  v.data.module   
    end
    
end



function trainBatch(inputsCPU,class)
   local nextF=1
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()
   local ferr = 0
   local f = 0
   
   local pb_grads = {[inputsCPU:size(1)-nextF+1] = torch.zeros(sizeBatch,64):cuda()}
   local rnn_state = {[0]= init_state_global}
   local PBs = {}
   local deltaPB = {}
   local features = {[0]=torch.zeros(sizeBatch, 64):cuda()}
   local outputs = {}
   local rnn_inputs_back  = {}

   local c_grad = {[inputsCPU:size(1)-nextF] = torch.zeros(sizeBatch,64):cuda()}

   model:zeroGradParameters()

   for t=1,inputsCPU:size(1)-nextF do
	PBs[t] = PB[{{class}}]:clone()
	deltaPB[t] = torch.CudaTensor():resizeAs(PBs[t]):fill(0)
	labels:copy(inputsCPU[{{t+nextF}}]) 
	input:copy(inputsCPU[{{t}}])
        local input_mem_cell = get_input_mem_cell()
        local rnn_inputs = {input, PBs[t], input_mem_cell, features[t-1], unpack(rnn_state[t-1])} 
	local output = model:forward(rnn_inputs)
	ferr = ferr + criterion:forward(output[1], labels)
	rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t],init_state_model[i].output) end
	features[t] = output[2]

    end
    local PB_grad = torch.zeros(sizeBatch,64):cuda()
    for t=inputsCPU:size(1)-nextF,1,-1 do
	labels:copy(inputsCPU[{{t+nextF}}]) 
	input:copy(inputsCPU[{{t}}])
        local input_mem_cell = get_input_mem_cell()
        local rnn_inputs = {input,  PBs[t], input_mem_cell, features[t-1], unpack(rnn_state[t-1])} 
	local output = model:forward(rnn_inputs)
	criterion:forward(output[1], labels)	

	local criBack = criterion:backward(output[1], labels)	

        drnn_state = {criBack, c_grad[t]}

	local rnnBack = model:backward(rnn_inputs,drnn_state)
	pb_grads[t] =  module_PB.gradInput:clone()

	c_grad[t-1] = rnnBack[4]
    end
    
    
    ferr = ferr / inputsCPU:size(1)
    dW:div(inputsCPU:size(1)-nextF)



    local eval_RNN = function(x)			
		return f, dW
    end
    optim.sgd(eval_RNN, w, optimStateIn2H)


   PB[{{class}}]:add(pb_grads[1]:mul(PB_learningrate))


   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + ferr
   --print(('Epoch: [%d][%d/%d]\tTime %.3f  Ferr %.4f LR %.5f - %.5f DataLoadingTime %.3f'):format(
   --       epoch, batchNumber, nEpochs, timer:time().real, ferr, 
   --      optimStateIn2H.learningRate,PB_learningrate, dataLoadingTime))
   
   dataTimer:reset()
end


function validBatch(inputsCPU,class,subject)
   local nextF=1
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()
   PB_temp = torch.zeros(1,64):cuda()
   local ferr = 0
   local f = 0
   
   local pb_grads = {[inputsCPU:size(1)-nextF+1] = torch.zeros(sizeBatch,64):cuda()}
   

   local rnn_state = {[0]= init_state_global}

   local PBs = {}
   local deltaPB = {}
   local features = {[0]=torch.zeros(sizeBatch, 64):cuda()}
   local outputs = {}
   local rnn_inputs_back  = {}

   local c_grad = {[inputsCPU:size(1)-nextF] = torch.zeros(sizeBatch,64):cuda()}
   local e_grad = torch.zeros(sizeBatch,4*68):cuda()

   model:zeroGradParameters()

   for t=1,inputsCPU:size(1)-nextF do
	PBs[t] = PB_temp:clone()
	deltaPB[t] = torch.CudaTensor():resizeAs(PBs[t]):fill(0)

	labels:copy(inputsCPU[{{t+nextF}}]) 
	input:copy(inputsCPU[{{t}}])

        local input_mem_cell = get_input_mem_cell()
        local rnn_inputs = {input, PBs[t], input_mem_cell, features[t-1], unpack(rnn_state[t-1])} 
	local output = model:forward(rnn_inputs)
	ferr = ferr + criterion:forward(output[1], labels)
	rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t],init_state_model[i].output) end
	features[t] = output[2]

    end
    local PB_grad = torch.zeros(sizeBatch,64):cuda()
    local count = 1
    for t=inputsCPU:size(1)-nextF,1,-1 do
	labels:copy(inputsCPU[{{t+nextF}}]) 
	input:copy(inputsCPU[{{t}}])
        local input_mem_cell = get_input_mem_cell()
        local rnn_inputs = {input,  PBs[t], input_mem_cell, features[t-1], unpack(rnn_state[t-1])} 
	local output = model:forward(rnn_inputs)
	criterion:forward(output[1], labels)	

	local criBack = criterion:backward(output[1], labels)	

        drnn_state = {criBack, c_grad[t]}

	local rnnBack = model:backward(rnn_inputs,drnn_state)
	pb_grads[t] =  module_PB.gradInput:clone()
	c_grad[t-1] = rnnBack[4]	
   end
   PB_temp:add(pb_grads[1]:mul(0.9))
   ferr = ferr / inputsCPU:size(1)
   PB_temp = PB_temp:float()
   ---------------Make folder ----------------------
   if os.execute('[ -e ./results_' .. opt.facial ..' ]') == nil then
   	os.execute('mkdir ./results_'.. opt.facial)
   end
   npy4th.savenpy('./results_'.. opt.facial .. '/' .. subject .. '_' .. classTrain[class] ..  '.npy', PB_temp)   
   -------------------------------------------------
end






