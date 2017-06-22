require 'nn'
require 'nngraph'
function lstm(h_t, h_d, prev_c, rnn_size)
  local all_input_sums = nn.CAddTable()({h_t, h_d})
  local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
  -- decode the gates
  local in_gate = nn.Sigmoid()(n1)
  local forget_gate = nn.Sigmoid()(n2)
  local out_gate = nn.Sigmoid()(n3)
  -- decode the write inputs
  local in_transform = nn.Tanh()(n4)
  -- perform the LSTM update
  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })
  -- gated cells form the output
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end

local GridLSTM = {}
function GridLSTM.grid_lstm(inputs,input_size, rnn_size, n, dropout, should_tie_weights)
  dropout = dropout or 0 


  local shared_weights
  if should_tie_weights == 1 then shared_weights = {nn.Linear(rnn_size, 4 * rnn_size), nn.Linear(rnn_size, 4 * rnn_size)} end

  local outputs_t = {} -- Outputs being handed to the next time step along the time dimension
  local outputs_d = {} -- Outputs being handed from one layer to the next along the depth dimension

  for L = 1,n do
    -- Take hidden and memory cell from previous time steps
    local prev_c_t = inputs[L*2+1]
    local prev_h_t = inputs[L*2+2]

    if L == 1 then
      -- We're in the first layer
      prev_c_d = inputs[1] -- input_c_d: the starting depth dimension memory cell, just a zero vec.
      prev_h_d = nn.Linear(input_size,rnn_size)(inputs[2]) -- input_h_d: the starting depth dimension hidden state. We map a char into hidden space via a lookup table
    else 
      -- We're in the higher layers 2...N
      -- Take hidden and memory cell from layers below
      prev_c_d = outputs_d[((L-1)*2)-1]
      prev_h_d = outputs_d[((L-1)*2)]
      if dropout > 0 then prev_h_d = nn.Dropout(dropout)(prev_h_d):annotate{name='drop_' .. L} end -- apply dropout, if any
    end

    -- Evaluate the input sums at once for efficiency
    local t2h_t = nn.Linear(rnn_size, 4 * rnn_size):init('bias', nninit.constant, 1)(prev_h_t):annotate{name='i2h_'..L}
    local d2h_t = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_d):annotate{name='h2h_'..L}
    
    -- Get transformed memory and hidden states pointing in the time direction first
    local next_c_t, next_h_t = lstm(t2h_t, d2h_t, prev_c_t, rnn_size)

    next_c_t:annotate{name='c_t_' .. L}
    next_h_t:annotate{name='h_t_' .. L}

    -- Pass memory cell and hidden state to next timestep
    table.insert(outputs_t, next_c_t)
    table.insert(outputs_t, next_h_t)

    -- Evaluate the input sums at once for efficiency
    local t2h_d = nn.Linear(rnn_size, 4 * rnn_size):init('bias', nninit.constant, 1)(next_h_t):annotate{name='i2h_'..L}
    local d2h_d = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_d):annotate{name='h2h_'..L}

    if should_tie_weights == 1 then
      print("tying weights along the depth dimension")
      t2h_d.data.module:share(shared_weights[1], 'weight', 'bias', 'gradWeight', 'gradBias')
      d2h_d.data.module:share(shared_weights[2], 'weight', 'bias', 'gradWeight', 'gradBias')
    end
    
    local next_c_d, next_h_d = lstm(t2h_d, d2h_d, prev_c_d, rnn_size)

    -- Pass the depth dimension memory cell and hidden state to layer above
    table.insert(outputs_d, next_c_d)
    table.insert(outputs_d, next_h_d)
  end

  -- set up the decoder
  local top_h = outputs_d[#outputs_d]:annotate{name='top_h'}
  local top_h = nn.ReLU(true)(top_h)
  local top_d = outputs_d[#outputs_d -1]:annotate{name='top_d'}
  --local proj_d = nn.ReLU()(nn.Linear(rnn_size,rnn_size - 2 * input_size)(top_d)):annotate{name='top_d'}
  --local top_c = outputs_d[#outputs_d-1]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  
  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  --local logsoft = nn.ReLU(true)(proj)
  table.insert(outputs_t, top_h)
  table.insert(outputs_t, proj)

  --return nn.gModule(inputs, outputs_t)

   return outputs_t
end

return GridLSTM
