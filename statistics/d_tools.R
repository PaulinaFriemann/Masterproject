require(readr)
require(ggplot2)
require(tidyverse)
require(Hmisc)
require(sjmisc)

fun_last_steps <- function(r) {
  # ranks at the last step
  last_t <- r %>% dplyr::group_by(run) %>% dplyr::summarise(t = max(t))
  last_t <- r %>% semi_join(last_t, by = c("run", "t"))
  last_t
}

true_model_ps <- function(dat) {
  values <- c()
  for (i in 1:nrow(dat)){
    # for every row, column of true model is value
    values[i] <- dat[[i, paste("p", dat$true_model[[i]], sep="_")]]
  }
  values
}