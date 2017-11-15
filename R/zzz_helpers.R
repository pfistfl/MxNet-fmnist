# Takes as input a matrix
# Returns the original data and the preprocessed data
rotate_data = function(x, do = "rotate") {
  # This function always returns the original x and x turned upside down
  # It thus effectively doubles the size
  switch(do, 
         rotate = {
           # rotate x by between 5 and 25 degrees
           rbind(x, t(apply(x, 1, random_rotate)))
         },
         mirror = {
           # mirror image vertically
           rbind(x, t(apply(x, 1, mirror_image)))
         },
         flip = {
           # flip x (turn upside down)
           n = sqrt(ncol(x))
           x_flip = order = rev(seq_len(n^2))
           rbind(x, x[, order])
         }, 
         transpose = {
           # transpose by 90 degrees
           order = as.integer(t(matrix(seq_len(n^2), nrow = n, ncol = n)))
           rbind(x, x[, order])
         },
         noise = {
           # put small noise on images
           rbind(x, noise_image(x))
         }
         )
}

random_rotate = function(x, min = 5, max = 25) {
  rot = sample(c(runif(1, -max, -min), runif(1, min, max)), 1)
  r = rotate_xy(as.cimg(as.numeric(x), 28, 28), rot, 14, 14)
  as.numeric(r)
}

mirror_image = function(x) {
  as.numeric(mirror(as.cimg(as.numeric(x), 28, 28), "x"))
}

noise_image = function(x, rate = 0.05) {
  # Put runif noise on pictures (only nonzero values)
  x_noise =  t(apply(x, 1, function(x) {
    x[x > 0] = x[x > 0] + runif(length(x[x > 0]), min = -rate, max = rate)
    x[x < 0] = 0
    x[x > 1] = 1
    return(x)
    }))
}


plot_item = function(x, prd) {
  p = implot(as.cimg(as.numeric(x), 28, 28), text(14, 14, as.character(prd), col = "red"))
  plot(p, xlim = c(0,28), ann = FALSE, axes = FALSE)
  }

make_submission = function(prediction, id, subm = "noname") {
  if (is.matrix(prediction)) {
    prediction = apply(prediction, 1, which.max) - 1
  }
  submission = data.frame(id = id, y = prediction)
  fwrite(submission, file = paste0("submissions/Submission_", subm, ".csv"))
}

# # FIXME: This is suboptimal
csv_to_array = function(path = "data/test.csv") {
  tst = fread(path)
  data = array(t(tst[,-1] / 255L), dim = c(28, 28, 1, nrow(tst)))
  list(data = data, id = tst$id) 
}

plot_items = function(x, prd) {
  dim = ceiling(sqrt(nrow(x)))
  par(mfrow = c(dim, dim))
  for (i in seq_len(nrow(x))) {
    plot_item(x[i,], prd[i])
  }
  par(mfrow = c(1,1))
}
