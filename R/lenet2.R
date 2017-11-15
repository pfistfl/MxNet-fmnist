LeNet2 = function() {
  # input
  data = mx.symbol.Variable('data')
  # first conv
  conv1 = mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter= 32, pad = c(2,2))
  bn1 = mx.symbol.BatchNorm(data=conv1, eps = 10^-4)
  tanh1 = mx.symbol.Activation(data=bn1, act_type="relu")
  pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=c(2,2), stride=c(2,2))
  # second conv
  conv2 = mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=64, pad = c(2, 2))
  bn2 = mx.symbol.BatchNorm(data=conv2, eps = 10^-4)
  tanh2 = mx.symbol.Activation(data=bn2, act_type="relu")
  pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=c(2,2), stride=c(2,2))
  # third conv
  conv3 = mx.symbol.Convolution(data=pool2, kernel=c(5,5), num_filter=128, pad = c(2, 2))
  bn3 = mx.symbol.BatchNorm(data=conv3, eps = 10^-4)
  tanh3 = mx.symbol.Activation(data=bn3, act_type="relu")
  pool3 = mx.symbol.Pooling(data=tanh3, pool_type="max", kernel=c(2,2), stride=c(2,2))
  # first fullc
  flatten = mx.symbol.Flatten(data=pool3)
  fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden = 512)
  tanh4 = mx.symbol.Activation(data=fc1, act_type="relu")
  drop1 = mx.symbol.Dropout(tanh4, p = 0.5)
  # second fullc
  fc2 = mx.symbol.FullyConnected(data=drop1, num_hidden=512)
  tanh5 = mx.symbol.Activation(data=fc2, act_type="relu")
  drop2 = mx.symbol.Dropout(tanh5, p = 0.5)
  # softmax
  fc3 = mx.symbol.FullyConnected(data=drop2, num_hidden=10)
  softmax = mx.symbol.SoftmaxOutput(data=fc3)
  return(softmax)
}


if (FALSE) {
  # Faster: Read first and make a Train / Eval Iterator
  x.train = fread("data/train_mnr_x_prep.csv")
  y.train = fread("data/train_mnr_y_prep.csv")
  train.iter = mx.io.arrayiter(
    data = array(t(x.train), dim = c(28, 28, 1, nrow(x.train))),
    label = y.train$V1,
    batch.size = 128,
    shuffle = TRUE)
  
  x.eval = fread("data/eval_x_prep.csv")
  y.eval = fread("data/eval_y_prep.csv")
  eval.iter = mx.io.arrayiter(
    data = array(t(x.eval), dim = c(28, 28, 1, nrow(x.eval))),
    label = as.numeric(y.eval$V1),
    batch.size = 128,
    shuffle = FALSE)
  
  # Train Model
  mx.set.seed(1244)
  logger = mx.metric.logger$new()
  architecture = LeNet2()
  model = mx.model.FeedForward.create(
    architecture,
    X = train.iter,
    eval.data = eval.iter,
    ctx = mx.gpu(),
    optimizer = "adam",
    eps = 10^-8, beta1 = 0.9, beta2 = 0.99,
    learning.rate = 0.001,
    wd = 0.0001,
    num.round = 30,
    eval.metric = mx.metric.accuracy,
    initializer = mx.init.Xavier(),
    batch.end.callback = mx.callback.log.train.metric(logger = logger, period = 1000))
  
  mx.model.save(model, "lenet2", iteration = 30)
  
  # Create submission file
  source("R/helpers.R")
  l = csv_to_array(path = "data/test.csv")
  prd = t(predict(model, l$data))
  make_submission(prd, l$id, subm = "LeNet2_40iter")
}
