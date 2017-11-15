LeNet = function(num_classes = 10) {
  # input
  data = mx.symbol.Variable('data')
  # first conv
  conv1 = mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter= 8, pad = c(2, 2))
  tanh1 = mx.symbol.Activation(data=conv1, act_type="relu")
  pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=c(2,2), stride=c(2,2))
  # second conv
  conv2 = mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=16, pad = c(2, 2))
  tanh2 = mx.symbol.Activation(data=conv2, act_type="relu")
  pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=c(2,2), stride=c(2,2))
  # first fullc
  flatten = mx.symbol.Flatten(data=pool2)
  fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden = 256)
  tanh3 = mx.symbol.Activation(data=fc1, act_type="relu")
  # second fullc
  fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
  # loss
  softmax = mx.symbol.SoftmaxOutput(data=fc2)
  return(softmax)
}