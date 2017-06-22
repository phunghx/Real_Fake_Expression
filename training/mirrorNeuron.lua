require 'nn'
require 'cunn'
require 'nngraph'
nninit = require 'nninit'
GridLSTM = paths.dofile('GridLSTM.lua')
function generateModel()
	--- input,  C, H, {C, H} * 4, PB
        local input_size = 40*2
        local nLayer = 4
	local lstm_size = 512
	x = nn.Identity()()
	PB = nn.Identity()()
	C1 = nn.Identity()()
	H1 = nn.Identity()()

	inputRNN = {C1, H1}
	input = {x,PB,C1,H1}
	for i=1,nLayer do
		local x1 = nn.Identity()()
		local x2 = nn.Identity()()
		table.insert(input, x1)
		table.insert(input, x2)
		table.insert(inputRNN, x1)
		table.insert(inputRNN, x2)
	end
	local rnnOut = GridLSTM.grid_lstm(inputRNN,64,lstm_size,nLayer,0,1)
	local hPB1 = nn.Linear(64,lstm_size)(PB):annotate{name = 'parametric_bias'}
	hPB = hPB1 - nn.ReLU(true)
	local hInput = x - nn.Linear(input_size,lstm_size) - nn.ReLU(true)
	hFinal = {hPB, rnnOut[nLayer*2+1],hInput} - nn.CAddTable(1,1) 
	Ah = hFinal - nn.Linear(lstm_size,input_size) - nn.Tanh(true)
	output = {Ah,rnnOut[nLayer*2+2]}
	model = nn.gModule(input,output)
	return model
end
