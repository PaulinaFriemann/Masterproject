require(readr)
require(ggplot2)
require(tidyverse)
require(Hmisc)
require(sjmisc)

dat <- read_csv("~/Uni/masterprojekt/masterproject/results/model_pred_convex.csv",
                col_types = cols(t = col_integer(), run=col_integer()))

values <- c()
for (i in 1:nrow(dat)){
  # for every row, column of true model is value
  values[i] <- dat[[i, paste("p", dat$true_model[[i]], sep="_")]]
}
dat$p_true_model <- values
dat <- dat[dat$p_true_model != 0,]

# calculating ranks over time 

ranks <- function(df) {
  p_values <- df %>% select((which(colnames(df)=="t") + 1):(ncol(df)))
  p_values$tm_idx <- paste("p", df$true_model, sep="_")
  p_values$tm_idx <- apply(p_values["tm_idx"],1, function(x) which(colnames(p_values) == x[1]))
  
  ranks_ <- data.frame(run=df$run, t=df$t, tm=df$true_model)

  ranks_$rank <- apply(p_values, 1, function(x) which(sort(x[1:length(x)-1], decreasing=TRUE) == x[tail(x,n=1)])[1])
  # ranks_$rank <- ranks_$rank / ncol(p_values)

  ranks_
  }
r <- ranks(subset(dat, select=-c(p_true_model)))
r$t <- as.integer(r$t)
r$run <- as.integer(r$run)
#points(r$t, r$rank)

# plotting ranks over time
n_models <- ncol((dat %>% select((which(colnames(dat)=="t") + 1):(ncol(dat))))) -1
ggplot(r[r$t > 0,], aes(x=t, y=rank)) +
  geom_smooth(method="gam") + ggtitle("GAM of model ranks") +
  xlab("time step t") + ylab(paste("rank of the correct model\n from N=", n_models, " models", sep=""))

# ranks at the last step
last_t <- r %>% group_by(run) %>% summarise(t = max(t))
last_t <- r %>% semi_join(last_t, by = c("run", "t"))
boxplot(last_t$rank, main="Ranks of the correct model at the final step", ylab="rank")

# percentage of models that are <
all_p_at_last_t <- subset(dat, 
                          select=-c(width, height, obstacles, 
                                    true_goal, distractor, start, trajectory))%>%
  semi_join(last_t, by=c("run", "t"))

percentage_worse <- c()
for (i in 1:nrow(all_p_at_last_t)) {
  percentage_worse[i] <- sum(all_p_at_last_t[i,4:ncol(all_p_at_last_t)] < 
                               all_p_at_last_t[[i,"p_true_model"]]) + 1
  percentage_worse[i] <- percentage_worse[i] / (ncol(all_p_at_last_t) - 4)
}
all_p_at_last_t$percentage_worse <- percentage_worse
boxplot(all_p_at_last_t$percentage_worse)

# percentage of models that are < for only beta variation
percentage_worse <- c()
for (i in 1:nrow(all_p_at_last_t)) {
  name <- unlist(str_split(all_p_at_last_t$true_model[i], "_"))[c(1,2)]
  name <- paste(name, collapse="_")
  print(name)

  beta_variations <- all_p_at_last_t[ , grepl( name , names( all_p_at_last_t ) ) ]
  print(beta_variations)
  percentage_worse[i] <- sum(beta_variations[i,] < 
                               all_p_at_last_t[[i,"p_true_model"]])
  percentage_worse[i] <- percentage_worse[i] / (ncol(beta_variations))
}
all_p_at_last_t$percentage_worse <- percentage_worse
summary(all_p_at_last_t$percentage_worse)

# Deviation plus/minus from correct beta -> p

# accuracy of model prediction
last_t <- r %>% group_by(run) %>% summarise(t = max(t))
last_t <- dat %>% semi_join(last_t, by = c("run", "t"))
# overall

# State type

# Algo type

# beta

# maybe after 50%?