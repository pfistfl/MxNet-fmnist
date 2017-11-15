# Simple convnets with a single fc layer to the output
# After 40 iterations this net achieves:
# [40] Train-accuracy=0.958814900426743
# [40] Validation-accuracy=0.848056891025641

simple_convnet = function() {
  data = mx.symbol.Variable("data")
  # 2D Convolution + Tanh + Max Pooling
  conv1 = mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=64, pad = c(2,2))
  tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
  pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=c(2,2), stride=c(2,2))
  # FC Layer for Output
  fc1 = mx.symbol.FullyConnected(pool1, name = "fc2", num_hidden = 10)
  softmax = mx.symbol.SoftmaxOutput(fc1, name = "sm")
  return(softmax)
}

if (FALSE) {
  
  # CSV Iterator is very slow. Possibly use arrayIterator instead.
  train.iter = mx.io.CSVIter(data.csv = "data/train_x_prep.csv",
                             data.shape = c(28, 28, 1),
                             label.csv = "data/train_y_prep.csv",
                             label.shape = 1,
                             batch.size = 128L,
                             prefetch.buffer = 15L)
  eval.iter = mx.io.CSVIter(data.csv = "data/eval_x_prep.csv",
                             data.shape = c(28, 28, 1),
                             label.csv = "data/eval_y_prep.csv",
                             label.shape = 1,
                             batch.size = 128L,
                             prefetch.buffer = 15L)
  t = Sys.time()
  # Train Model
  mx.set.seed(1244)
  logger = mx.metric.logger$new()
  architecture = simple_convnet()
  model = mx.model.FeedForward.create(
    architecture,
    X = train.iter,
    eval.data = eval.iter,
    ctx = mx.gpu(),
    optimizer = "adam",
    eps = 10^-8, beta1 = 0.9, beta2 = 0.99,
    learning.rate = 0.001,
    wd = 0.0001,
    num.round = 40,
    eval.metric = mx.metric.accuracy,
    initializer = mx.init.Xavier(),
    batch.end.callback = mx.callback.log.train.metric(logger = logger, period = 100))
  sprintf("Took %s seconds!", Sys.time() - t)
  
  # Create submission file
  source("R/helpers.R")
  l = csv_to_array()
  prd = t(predict(model, l$data))
  make_submission(prd, l$id, subm = "simple_fullyconnected_40iters")
}
