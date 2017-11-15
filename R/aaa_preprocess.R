require("mxnet")
require("ggplot2")
require("reshape2")
require("data.table")
require("imager")
source("R/zzz_helpers.R")

trn = fread("data/train.csv")

# Train / Test Split -> Drop y -> Preprocess -> Write To CSV 
set.seed(4442L)
train.inds = sample(seq_len(nrow(trn)), 0.9 * nrow(trn))
x.train = as.matrix(trn[train.inds, -1] / 255L)
x.train = x.train %>%
  rotate_data(do = "mirror") %>%
  rotate_data(do = "noise") %>%
  rotate_data(do = "rotate")
# We have increased our data by 8 times, thus rep y col 8 times
y.train = rep(as.numeric(trn[train.inds, 1]$y), 8)

# Save to a .csv
fwrite(data.table(x.train), "kaggle/mnist_dlcourse/data/train_mnr_x_prep.csv", col.names = FALSE)
fwrite(data.table(y.train), "kaggle/mnist_dlcourse/data/train_mnr_y_prep.csv", row.names = FALSE, col.names = FALSE)

# Evaluation Data
x.eval = trn[-train.inds, -1] / 255
y.eval = as.numeric(trn[-train.inds, 1]$y)
fwrite(data.table(x.eval), "kaggle/mnist_dlcourse/data/eval_x_prep.csv", col.names = FALSE)
fwrite(data.table(y.eval), "kaggle/mnist_dlcourse/data/eval_y_prep.csv", row.names = FALSE, col.names = FALSE)

# Slow: CSV.Iterator (BUT: Data is not kept in Memory)
# train.iter = mx.io.CSVIter(data.csv = "kaggle/mnist_dlcourse/data/train_x_prep.csv",
#                            data.shape = c(28, 28, 1),
#                            label.csv = "kaggle/mnist_dlcourse/data/train_y_prep.csv",
#                            label.shape = 1,
#                            batch.size = 128L)
# eval.iter = mx.io.CSVIter(data.csv = "kaggle/mnist_dlcourse/data/eval_x_prep.csv",
#                            data.shape = c(28, 28, 1),
#                            label.csv = "kaggle/mnist_dlcourse/data/eval_y_prep.csv",
#                            label.shape = 1,
#                            batch.size = 256L)

# Faster: Read data into memory first and make a Train/ Eval Iterator
x.train = fread("kaggle/mnist_dlcourse/data/train_mnr_x_prep.csv")
y.train = fread("kaggle/mnist_dlcourse/data/train_mnr_y_prep.csv")
train.iter = mx.io.arrayiter(
  data = array(t(x.train), dim = c(28, 28, 1, nrow(x.train))),
  label = y.train,
  batch.size = 128,
  shuffle = TRUE)

x.eval = fread("kaggle/mnist_dlcourse/data/eval_x_prep.csv")
y.eval = fread("kaggle/mnist_dlcourse/data/eval_y_prep.csv")
eval.iter = mx.io.arrayiter(
  data = array(t(x.eval), dim = c(28, 28, 1, nrow(x.eval))),
  label = y.eval,
  batch.size = 128,
  shuffle = FALSE)
