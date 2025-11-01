#' Plot partial dependence probabilities
#'
#'@param x iucnn_get_pdp output. A list of class icunn_pdp named according to
#'the feature and the partial dependence probabilities for the IUCN category.
#'See \code{\link{iucnn_get_pdp}}
#'@param ask logical (default = FALSE). Indicates if the user is prompted before
#'a new plot will be displayed.
#'@param features (default=NULL). Optional vector of integers specifying which
#'feature's PD should be plotted, e.g. c(1, 3).
#'@param uncertainty logical (default = TRUE). Should dropout uncertainty be
#'displayed?
#'@param order_categorical (default = TRUE). Should categorical features be
#'ordered so that extinction risk increased along the x-axis or should they be
#'displayed in the order given by x.
#'@param col (default = NULL). Custom colors for IUCNN categories.
#'@param ... further graphical parameter for par, e.g. different margins with
#'mar = c(10, 4, 0.5, 0.5)
#'
#' @export
#'
plot.iucnn_pdp <- function(x,
                           ask = FALSE,
                           features = NULL,
                           uncertainty = TRUE,
                           order_categorical = TRUE,
                           col = NULL, ...) {
  if (is.null(features)) {
    features <- 1:length(x)
  }
  feature_names <- names(x)

  num_cats <- ncol(x[[features[1]]]$pdp)

  if (is.null(col)) {
    col <- c("#468351", "#BBD25B", "#F4EB5A", "#EDAA4C", "#DA4741")
    if (num_cats == 2) {
      col <- col[c(2, 5)]
    }
  }

  if (uncertainty) {
    uncertainty <- length(x[[1]]) == 4
  }

  for (fe in 1:length(features)) {
    cont_feature <- is.numeric(x[[features[fe]]]$feature[, 1])
    x_fe <- x[[fe]]

    ask_plot <- FALSE
    if (ask && fe > 1) {
      ask_plot <- TRUE
    }

    if (cont_feature) {
      par(las = 1, ask = ask_plot, ...)
      plot(0, 0, type = "n",
           ylim = c(0, 1), xlim = range(x_fe$feature),
           xlab = feature_names[fe],
           ylab = "Partial dependence probability",
           xaxs = "i", yaxs = "i")

      if (uncertainty) {
        polygon(x = c(x_fe$feature, rev(x_fe$feature)),
                y = c(x_fe$lwr[, 1], rep(0, nrow(x_fe$pdp))),
                border = NA, col = col[1])
        polygon(x = c(x_fe$feature, rev(x_fe$feature)),
                y = c(x_fe$lwr[, 1], rev(x_fe$upr[, 1])),
                border = NA, col = "grey")
        if (num_cats == 5) {
          for (i in 2:(num_cats - 1)) {
            polygon(x = c(x_fe$feature, rev(x_fe$feature)),
                    y = c(x_fe$lwr[, i], rev(x_fe$upr[, i - 1])),
                    border = NA, col = col[i])
            polygon(x = c(x_fe$feature, rev(x_fe$feature)),
                    y = c(x_fe$lwr[, i], rev(x_fe$upr[, i])),
                    border = NA, col = "grey")
          }
        }
        polygon(x = c(x_fe$feature, rev(x_fe$feature)),
                y = c(rep(1, nrow(x_fe$pdp)), rev(x_fe$upr[, num_cats - 1])),
                border = NA, col = col[num_cats])
        for (i in 1:(num_cats - 1)) {
          lines(x_fe$feature, x_fe$pdp[, i], col = col[i + 1], lwd = 3)
        }
      }

      else {
        polygon(x = c(x_fe$feature, rev(x_fe$feature)),
                y = c(x_fe$pdp[, 1], rep(0, nrow(x_fe$pdp))),
                border = NA, col = col[1])
        for (i in 2:num_cats) {
          polygon(x = c(x_fe$feature, rev(x_fe$feature)),
                  y = c(x_fe$pdp[, i], rev(x_fe$pdp[, i - 1])),
                  border = NA, col = col[i])
        }
      }
    }

    else {

      if (order_categorical) {
        x_fe <- order_cat_pdp(x_fe, num_cats, uncertainty)
      }

      par(las = 1, ask = ask_plot, ...)
      plot(0, 0, type = "n",
           ylim = c(0, 1), xlim = c(0, nrow(x_fe$feature)),
           xlab = "", ylab = "Partial dependence probability",
           xaxs = "i", yaxs = "i", xaxt = "n")
      par(las = 2)
      axis(side = 1, at = 1:nrow(x_fe$feature) - 0.5,
           labels = x_fe$feature[, 1])

      if (uncertainty) {
        for (j in 1:nrow(x_fe$feature)) {
          p <- x_fe$pdp[j, ]
          l <- x_fe$lwr[j, ]
          u <- x_fe$upr[j, ]
          rect(xleft = j - 1, xright = j, ybottom = 0, ytop = l[1],
               border = NA, col = col[1])
          rect(xleft = j - 1, xright = j, ybottom = l[1], ytop = u[1],
               border = NA, col = "grey")
          segments(x0 = j - 1, x1 = j, y0 = p[1], y1 = p[1],
                   col = col[2], lwd = 3)
          if (num_cats == 5) {
            for (k in 2:(num_cats - 1)) {
              rect(xleft = j - 1, xright = j, ybottom = u[k - 1], ytop = l[k],
                   border = NA, col = col[k])
              rect(xleft = j - 1, xright = j, ybottom = u[k], ytop = l[k],
                   border = NA, col = "grey")
              segments(x0 = j - 1, x1 = j, y0 = p[k], y1 = p[k],
                       col = col[k + 1], lwd = 3)
            }
          }
          rect(xleft = j - 1, xright = j, ybottom = u[num_cats - 1], ytop = 1,
               border = NA, col = col[num_cats])
        }
      }
      else {
        for (j in 1:nrow(x_fe$feature)) {
          p <- c(0, x_fe$pdp[j, ])
          for (k in 2:(num_cats + 1)) {
            rect(xleft = j - 1, xright = j, ybottom = p[k - 1], ytop = p[k],
                 border = NA, col = col[k - 1])
          }
        }
      }

    }
  }
}


order_cat_pdp <- function(x_fe, num_cats, uncertainty) {
  if (num_cats == 5) {
    pca1 <- prcomp(scale(x_fe$pdp[, 1:4]))$x[, 1]
    ord <- order(pca1)
    a <- x_fe$pdp[ord, 1]
    if (a[1] < a[length(a)]) {
      ord <- rev(ord)
    }
  } else {
    ord <- order(x_fe$pdp[, 1])
  }
  x_fe$feature <- x_fe$feature[ord, , drop = FALSE]
  x_fe$pdp <- x_fe$pdp[ord, ]
  if (uncertainty) {
    x_fe$lwr <- x_fe$lwr[ord, ]
    x_fe$upr <- x_fe$upr[ord, ]
  }
  return(x_fe)
}
