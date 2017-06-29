require 'nn'
require 'cunn'
require 'nngraph'
require 'optim'
nninit = require 'nninit'
--GridLSTM = paths.dofile('GridLSTM.lua')

cmd = torch.CmdLine()
cmd:option('-facial', 'anger', 'facial emotion')
cmd:option('-data', 'data/shape', 'dataset path')
opt = cmd:parse(arg or {})

classes = {['anger']={'N2A','H2N2A'},
	   ['contentment']={'N2C','H2N2C'},
	   ['disgust']={'N2D','H2N2D'},
	   ['happy']={'N2H','S2N2H'},
	   ['sadness']={'N2S','H2N2S'},
	   ['surprise']={'N2Sur','D2N2Sur'}}

classTrain = classes[opt.facial]

print("======== Generate training on " .. opt.facial)

--paths.dofile('mirrorNeuron.lua')   

--model = generateModel()
--model = model:cuda()
--PB = torch.FloatTensor(2,64):fill(0):cuda()
model = torch.load(paths.concat('../' .. opt.facial, 'model.t7'))
PB = torch.load(paths.concat('../' .. opt.facial, 'PB_.t7'))
model = model:cuda()
PB = PB:cuda()

criterion = nn.MSECriterion():cuda()
collectgarbage()
paths.dofile('training_val.lua')
nEpochs=1
epoch = 1
for i=1,nEpochs do
	train(true)
	epoch = epoch + 1
	print("Epoch " .. epoch .. " " .. opt.facial)
end


