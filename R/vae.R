library("mxnet")
library("R6")
library("checkmate")

#' Variational Autoencoder for mxnet
#' 
#' Taken and adapted from (https://github.com/apache/incubator-mxnet/blob/master/example/vae/VAE.py).
#' @field n_latent Dimension of the latent space Z.
#' @field num_ecoder Number of hidden units in the encoder.
#'                   Decoder has the same size.
#' @field batch_size Mini batch size
#' @field learning_rate Learning rate during training
#' @field weight_decay Weight decay during training, for regularization of parameters
#' @field num_epoch Total number of training epoch
#' @field optimizer 'sgd' optimizer by default
VAE = R6Class("VAE", lock_objects = FALSE,
  public = list(
    Bernoulli = function(x_hat, loss_label) {
      mx.symbol.negative(mx.symbol.sum_axis(
        mx.symbol.broadcast_mul(loss_label, mx.symbol.log(x_hat)) +
          mx.symbol.broadcast_mul(1 - loss_label, mx.symbol.log(1-x_hat)
          ), axis=1))
    },
    initialize = function(x_train, x_valid = NULL, n_latent = 7, n_encoder = 128L,
      learning_rate = 0.001, weight_decay = 0.01, num_epoch = 100L, optimizer = "adam",
      likelihood = self$Bernoulli, batch_size = 128L, initializer = mx.init.Xavier()) {
      private$n_latent = n_latent
      private$n_encoder = n_encoder
      private$learning_rate = learning_rate
      private$weight_decay = weight_decay
      private$num_epoch = asInt(num_epoch)
      private$optimizer = optimizer
      private$initializer = initializer
      private$likelihood = likelihood
      private$batch_size = batch_size
      lst = self$train_vae(x_train, x_valid, batch_size, n_latent, n_encoder,
        learning_rate, weight_decay, num_epoch, optimizer,
        likelihood, initializer)
      private$model = lst$model
      private$symbol = lst$symbol
      private$logger = lst$logger
    },
    train_vae = function(x_train, x_valid, batch_size, n_latent, n_encoder, learning_rate,
      weight_decay, num_epoch, optimizer, likelihood, initializer) {
      
      # create data iterator to feed into NN
      N = nrow(x_train)
      features = ncol(x_train)
      
      if(!(class(x_train) == "Rcpp_MXNativeDataIter"))
        nd_iter = mx.io.arrayiter(
          data = t(x_train),
          label = t(x_train),
          batch.size = batch_size)
      else
        nd_iter = x_train
      
      if (!is.null(x_valid) & (!(class(x_valid) == "Rcpp_MXNativeDataIter")))
        nd_iter_val = mx.io.arrayiter(
          data = t(x_valid),
          label = t(x_valid),
          batch.size = batch_size)
      else
        nd_iter_val = NULL
      
      # Data Input / Output
      input = mx.symbol.Variable('data')
      loss_label = mx.symbol.Variable('label')
      
      # Encoder
    
      for (num in n_encoder)
        input = private$makeLayer(input, type = "encode", num)
      
      # Reduce to n_latent
      mu = mx.symbol.FullyConnected(data=input, name="mu", num_hidden = n_latent)
      logvar = mx.symbol.FullyConnected(data=input, name="logvar", num_hidden = n_latent)
      
      # Latent manifold
      z = mu + mx.symbol.broadcast_mul(
        mx.symbol.exp(0.5 * logvar),
        mx.symbol.random_normal(loc=0, scale=1, shape = c(n_latent, batch_size))
        )
      
      # Decoder
      output = z
      for (num in c(rev(n_encoder), features))
        output = private$makeLayer(output, type = "decode", num)
      
      # Compute KL Divergence and minus ELBO to minimize
      # FIXME: Check KL Divergence
      KL = -0.5 * mx.symbol.sum_axis(1 + logvar - mu^2 - mx.symbol.exp(logvar), axis=1)
      loss = likelihood(output, loss_label) + KL
      output = mx.symbol.MakeLoss(mx.symbol.sum(loss), name = 'vae_loss')
      
      logger = mx.metric.logger$new()
      model = mx.model.FeedForward.create(
        symbol = output, X = nd_iter, eval.data = nd_iter_val,
        num.round = num_epoch,
        ctx = mx.cpu(),
        optimizer = optimizer,
        initializer = initializer,
        learning.rate = learning_rate,
        wd = weight_decay,
        eps = 10^-8, beta1 = 0.9, beta2 = 0.99,
        epoch.end.callback = mx.callback.log.train.metric(logger = logger, period = 1L),
        mx.callback.log.train.metric(logger = logger, period = 1L)
      )
      return(list("model" = model, "symbol" = output, "logger" = logger))
    },
    get_model = function(){private$model}, 
    get_symbol = function(){private$symbol},
    get_logger = function(){private$logger}
  ),
  private = list(
    makeLayer = function(input, type = "encode", num, act_type = "tanh") {
      if (type == "decode")
        act_type = "sigmoid"
      fc = mx.symbol.FullyConnected(data = input, name = paste0("fc_", type, "_", num), num_hidden = num)
      output = mx.symbol.Activation(data = fc, name = paste0("act_", type, "_", num), act_type = act_type)
      return(output)
    }
  )
)

train.iter = mx.io.CSVIter(data.csv = "../kaggle/mnist_dlcourse/data/train_x.csv",
  data.shape = c(28*28, 1),
  label.csv = "../kaggle/mnist_dlcourse/data/train_x.csv",
  label.shape = c(28*28, 1),
  batch.size = 128L)

library(data.table)
x.train = fread("../kaggle/mnist_dlcourse/data/train_x.csv")
x.train = data.matrix(x.train)
vae = VAE$new(x_train = x.train, n_latent = 32L, n_encoder = c(128L, 64L), num_epoch = 1)
graph.viz(vae$get_symbol())
vae$get_logger()
