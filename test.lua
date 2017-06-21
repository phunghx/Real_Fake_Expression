
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
--require 'optim'
--csvigo = require 'csvigo'
--local dbg = require("debugger")
npy4th = require 'npy4th'
nn.DataParallelTable.deserializeNGPUs = 1


FacialExpression = {'anger','contentment','disgust','happy','sadness','surprise'}
folder_files = {['anger']='ANGER',['contentment']='CONTENTMENT',['disgust']='DISGUST',['happy']='HAPPINESS',['sadness']='SADNESS',['surprise']='SURPRISE'}

local nextF=1
local sizeBatch = 1
local input_size = 40*2
local lstm_size = 512
init_state = {}
local h_init = torch.zeros(sizeBatch, lstm_size)
h_init = h_init:cuda()
for L=1, 4 do --num layer = 4, change in model also
   table.insert(init_state, h_init:clone())
   table.insert(init_state, h_init:clone())
end
input = torch.CudaTensor(sizeBatch,input_size)
labels = torch.CudaTensor(sizeBatch,input_size)

function get_input_mem_cell()
    local input_mem_cell = torch.zeros(sizeBatch, lstm_size)
    input_mem_cell = input_mem_cell:float():cuda()
    return input_mem_cell
end

trainHook = function(input) -- sequenc x 1 x 128 x 128
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
   out:mul(2):add(-1)
   return out
end

function getSubjectTest(folderPath)

   local maxlength = 0
   for file in paths.iterfiles('data/testshape_v1/' .. folderPath) do
	maxlength = maxlength + 1
   end

   local data_full = torch.FloatTensor(maxlength-1,1,68*2)
   for j=2,maxlength do
	local shape = npy4th.loadnpy('data/testshape_v1/' .. folderPath .. string.format('/%04d.npy',j))
	data_full[j-1][1]:copy(shape)
	
   end   
   return trainHook(data_full)
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
function getPBVector(model, module_PB,init_state_model, inputsCPU, subject, id)
       cutorch.synchronize()
       collectgarbage()
       local PB_temp = torch.zeros(1,64):cuda()

       local pb_grads = {[inputsCPU:size(1)-nextF+1] = torch.zeros(sizeBatch,64):cuda()}
       local rnn_state = {[0]= clone_list(init_state)}
       local PBs = {}
       local deltaPB = {}
       local features = {[0]=torch.zeros(sizeBatch, 64):cuda()}
       local outputs = {}
       local rnn_inputs_back  = {}
       local ferr = 0
       local c_grad = {[inputsCPU:size(1)-nextF] = torch.zeros(sizeBatch,64):cuda()}

       model:zeroGradParameters()
       criterion = nn.MSECriterion():cuda()
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
       cutorch.synchronize()
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
		--if (batchNumber==15) then dbg() end
		local rnnBack = model:backward(rnn_inputs,drnn_state)
		pb_grads[t] =  module_PB.gradInput:clone()
		c_grad[t-1] = rnnBack[4]
	
		--cutorch.synchronize()
		--collectgarbage()
       end
       cutorch.synchronize()
       PB_temp:add(pb_grads[1]:mul(0.9))
       ferr = ferr / inputsCPU:size(1)
       PB_temp = PB_temp:float()
       ---------------Make folder ----------------------
       npy4th.savenpy(subject .. '/data/' .. id ..  '_PB.npy', PB_temp:squeeze())
       collectgarbage()

end


for k, facial in pairs(FacialExpression) do
	model = torch.load(facial .. '/model.t7')
	model:evaluate()
	local init_state_model = {}
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
	for dir in paths.iterdirs("data/testshape") do
		
		Express = dir:split('_')[2]
		if Express == folder_files[facial] then
		    inputs = getSubjectTest(dir)
		    getPBVector(model, module_PB,init_state_model, inputs, facial, dir)
		end
	end
end



