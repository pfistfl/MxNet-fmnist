# Unfinished implementation of capsnet in R-Keras
# Code is mostly obtained/adapted from AaronLeong/CapsNet_Mxnet
################################################################################
stop("This implementation is unfinished and does currently misses most of the code")
################################################################################
library("keras")

layer_primaryCaps = function(inputs, dim_vector, n_channels, kernel_size, strides, padding) {

    # Apply Conv2D `n_channels` times and concatenate all capsules
    # :param inputs: 4D tensor, shape=[None, width, height, channels]
    # :param dim_vector: the dim of the output vector of capsule
    # :param n_channels: the number of types of capsules
    # :return: output tensor, shape=[None, num_capsule, dim_vector]

  input %>% 
    layer_conv_2d(filters=dim_vector*n_channels, kernel_size=kernel_size, strides=strides,
                         padding=padding, name='primarycap_conv2d') %>%
    layer_reshape(target_shape=c(-1, dim_vector), name='primarycap_reshape') %>%
    layer_lambda(squash, name='primarycap_squash')
}


squash = function(vectors, axis=-1) {
    # The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    # :param vectors: some vectors to be squashed, N-dim tensor
    # :param axis: the axis to squash
    # :return: a Tensor with same shape as input vectors

  K$sum(K$sqsquare(vectors), axis, keepdims=TRUE)
  scale = s_squared_norm / (1 + s_squared_norm) / K$sqrt(s_squared_norm + K$epsilon())
  scale * vectors
}

Length = list(
  # Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss
  # inputs: shape=[dim_1, ..., dim_{n-1}, dim_n]
  # output: shape=[dim_1, ..., dim_{n-1}]
  
 call = function(inputs){
  K$sqrt(K$sum(K$square(inputs), -1))
 }

 compute_output_shape = function(input_shape){
  input_shape[,-1]
 }
)


# Mask = list(
#   # Mask a Tensor with shape=[None, d1, d2] by the max value in axis=1.
#   # Output shape: [None, d2]
#   call = function(inputs){
#   # use true label to select target capsule, shape=[batch_size, num_capsule]
#   if (is.list(inputs))  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
#     inputs, mask = inputs
#     
#   } else {
#     # if no true label, mask by the max length of vectors of capsules
#     x = inputs
#     # Enlarge the range of values in x to make max(new_x)=1 and others < 0
#     x = (x - K.max(x, 1, True)) / K.epsilon() + 1 
#     mask = K.clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0
# 
#     # masked inputs, shape = [batch_size, dim_vector]
#     inputs_masked = K.batch_dot(inputs, mask, [1, 1])
# })

  # compute_output_shape = function(input_shape) {
  # if (is.numeric(input_shape)) label provided
  # tuple([None, input_shape[0][-1]])
  # } else {
  # return tuple([None, input_shape[-1]])
  # }


K <- backend()
model <- keras_model_sequential() 
model %>% 
  layer_conv_2d(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')
