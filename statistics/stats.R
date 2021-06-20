source("d_tools.R")
require(Kendall)

# load data
dat <- read_csv("~/Uni/masterprojekt/masterproject/results/model_pred_noobstacles.csv",
                col_types = cols(t = col_integer(), run=col_integer()))
n_models <- ncol((dat %>% select((which(colnames(dat)=="t") + 1):(ncol(dat)))))
p_true_models <- true_model_ps(dat)
dat$p_true_model <- p_true_models
dat <- dat[dat$p_true_model != 0,] # TODO somehow it crashed for a few trials

# split the "true model" column into 3
dat <- dat %>% separate(true_model, c("state","algorithm","beta"), "_")
dat$state <- as.factor(dat$state)
dat$algorithm <- as.factor(dat$algorithm)
dat$beta <- as.numeric(dat$beta)
dat$state <- plyr::revalue(dat$state, c("DistanceDirectionState"="Vector","GPSState"="Grid"))


##### stats

baseline <- function(yintercept=(1/n_models)) {
  geom_hline(yintercept=yintercept, linetype="dashed", color = "red", show.legend = TRUE)
}

baseline_text <- function(x_, y_=.04) {
  annotate(geom="text", x=x_, y=y_, label="Baseline",color="red", size=3.5)
}

dat$run <- as.factor(dat$run)
ggplot(dat[dat$t > 0,], aes(x=t, y=p_true_model)) +
  geom_line()

# p of true model over time
ggplot(dat[dat$t > 0,], aes(x=t, y=p_true_model)) +
  geom_line(colour=run) +
  geom_smooth() + ggtitle("Model likelihood over time") +
  baseline() + baseline_text(5, y_=.07) +
  xlab("time step t") + ylab(paste("likelihood of the correct model\n with N=", n_models, " models", sep=""))

# last time step
#dat <- subset(dat, select=-c(p_true_model))
last_steps <- fun_last_steps(dat)
last_step_tm <- last_steps$p_true_model
summary(subset(last_steps, select=-c(p_true_model))$t)


# p of true model at last time step
# x: state y: p fill: algo
ggplot(last_steps, aes(x=state, y=p_true_model, fill=algorithm)) +
  geom_boxplot() + ggtitle("Model likelihood at last step") +
  geom_dotplot(binaxis='y', stackdir='center',
               position=position_dodge(.75)) +
  xlab("Representation Type") + ylab("p value of the correct model") +
  labs(fill="Algorithm Type")

# x: algo y: p fill: state
ggplot(last_steps, aes(x=algorithm, y=p_true_model, fill=state)) +
  geom_boxplot() + ggtitle("Model likelihood at last step") +
  geom_dotplot(binaxis='y', stackdir='center',
               position=position_dodge(.75)) +
  xlab("Algorithm Type") + ylab("p value of the correct model") +
  labs(fill="Representation Type")


# x: beta y: p true model
# vector vs grid
ggplot(subset(last_steps, algorithm=="Optimal"), aes(beta, p_true_model, colour=state)) +
  geom_smooth(se=FALSE) + geom_point() +
  baseline() + baseline_text(.5)

# greedy vs optimal
ggplot(subset(last_steps, state=="Vector"), aes(beta, p_true_model, colour=algorithm)) +
  geom_smooth(se=FALSE) + geom_point() +
  baseline() + baseline_text(.5)

# x: beta y: sum(p values) for greedy vs optimal
temp <- last_steps
temp$Optimal <- rowSums(select(last_steps, contains("Optimal")))
temp$Greedy <- rowSums(select(last_steps, contains("Greedy")))
temp$Grid <- rowSums(select(last_steps, contains("GPS")))
temp$Vector <- rowSums(select(last_steps, contains("DistanceDirection")))
# todo figure out what plot should look like
# color: ground truth
temp <- select(temp, c("run", "state", "algorithm", "beta", 
                       "t", "Optimal", "Greedy", "Grid", "Vector"))
# make columns for sum of p values for the correct algorithm type / state type
values <- c()
for (i in 1:nrow(temp)){
  # for every row, select column based on value of algorithm
  values[i] <- temp[[i, temp$algorithm[[i]]]]
}
temp$correct_algo <- values
values <- c()
for (i in 1:nrow(temp)){
  # for every row, select column based on value of algorithm
  values[i] <- temp[[i, temp$state[[i]]]]
}
temp$correct_state <- values

# TODO plot sucks
temp %>% group_by(beta) %>% summarise(algo_det=mean(correct_algo)) %>%
  ggplot(aes(beta, algo_det)) +
  geom_smooth(method="lm", formula=y~I(x^3)+I(x^2))+ baseline(yintercept=.5) +
  baseline_text(4.5, y_=.53)

means_ <- temp %>% group_by(beta) %>% summarise(algo_det=mean(correct_algo))
a <- temp %>% full_join(means_)

a %>% ggplot(aes(beta, y=correct_algo)) +# method.args = list(family = "binomial"))+
  geom_line(aes(y=algo_det))+
  baseline(yintercept=.5) + baseline_text(4.5, y_=.53) +
  ylab("Sum of p values") + xlab(expression(beta)) +
  ggtitle(expression(paste("Likelihood of the correct algorithm")))
  
# number of correct identifications (>= .95)
sum(temp$correct_algo >= .95) / nrow(temp)
sum(temp$correct_state >= .95) / nrow(temp)

p_values <- function(df) {
  df %>%
    select((which(colnames(df)=="t") + 1):(ncol(df)))
}

# number of correct identifications overall
correct <- p_values(last_steps) %>% apply(1, function(x) max(x))
correct <- correct == last_steps$p_true_model
paste(sum(correct) / length(correct), "correct identifications including ties")
# number where tm has highest p value, but with ties
#p_values[correct,]
only_correct <- last_steps[correct,]
ties <- rowSums(p_values(only_correct) == only_correct$p_true_model)
#t(t(apply(m, 1, function(u) paste( names(which(u)), collapse=", " ))))
(p_values(only_correct) == only_correct$p_true_model) %>% apply(1, 
  function(u) paste( names(which(u)), collapse="," )
)
print("in all cases where the true model has the highest p,x
  there is a tie with the other state representation")

# tm better than .... % of models
perc_worse <- (n_models - rowSums(p_values(subset(last_steps, select=-c(p_true_model))) >
                      last_steps$p_true_model)) / n_models
summary(perc_worse)
paste("tm is in most cases among the models with the highest likelihood (mean:",
      mean(perc_worse), "std:",sd(perc_worse), ")")

# beta=0, beta=3, greedy vs optimal, vector vs grid -> p values
beta_2 <- last_steps[last_steps$beta==0,]
beta_2 <- subset(beta_2, select=-c(p_true_model))
beta_2 <- beta_2 %>% rename(tm_state=state, tm_algo=algorithm, tm_beta=beta)
beta_2 <- beta_2 %>% pivot_longer(
  cols = starts_with("p_"),
  names_to = c("state", "algo", "beta"),
  names_prefix = "p_",
  names_pattern = "(.*)_(.*)_(.*)",
  values_to = "p"
)
beta_2$beta <- as.double(beta_2$beta)
beta_2$state <- as.factor(beta_2$state)
beta_2$algo <- as.factor(beta_2$algo)

ggplot(beta_2, aes(colour=algo, y=p, x=tm_algo)) + geom_boxplot() +
  xlab("Algorithm of the true model") + labs(fill="Algorithm")
ggplot(beta_2[beta_2$tm_algo=="Greedy",], aes(y=p, x=algo)) + geom_boxplot() +
  xlab("Algorithm of the true model") + labs(fill="Algorithm")
# TODO test??
ggplot(beta_2, aes(y=p, x=tm_algo)) + geom_boxplot() +
  xlab("Algorithm of the true model") + labs(fill="Algorithm")
wilcox.test(beta_2[beta_2$tm_algo=="Greedy" & beta_2$algo=="Optimal",]$p,
            beta_2[beta_2$tm_algo=="Greedy" & beta_2$algo=="Greedy",]$p,
            paired=FALSE)


ggplot(beta_2, aes(y=p, x=beta, colour=algo)) + geom_smooth() +
  xlab(expression(beta)) + labs(colour="Algorithm") +
  ggtitle(expression(paste("Model likelihood for ", beta, "= ")))


long_last_t <- subset(last_steps, select=-c(p_true_model))
long_last_t <- long_last_t %>% rename(tm_state=state, tm_algo=algorithm, tm_beta=beta)
long_last_t <- long_last_t %>% pivot_longer(
  cols = starts_with("p_"),
  names_to = c("state", "algo", "beta"),
  names_prefix = "p_",
  names_pattern = "(.*)_(.*)_(.*)",
  values_to = "p"
)
long_last_t$beta <- as.factor(long_last_t$beta)
long_last_t$state <- as.factor(long_last_t$state)
long_last_t$algo <- as.factor(long_last_t$algo)

f <- function(x) {
  max(c(0.0,x))
}
long_last_t %>% ggplot(aes(x=tm_beta, y=p, colour=beta)) +
  geom_smooth(formula=y~x,span=.3,se=FALSE) +
  xlab(expression(paste(beta, " of the true model"))) +
  ggtitle(expression(paste("Recognition of ", beta, " values"))) +
  labs(colour=expression(beta))

long_last_t %>% group_by(algo) %>% summarise(sum_algo=sum(p)) %>%
  ggplot(aes(x=tm_beta, y=p, colour=sum_algo)) +
  geom_smooth(formula=y~x,span=.3,se=FALSE) +
  xlab(expression(paste(beta, " of the true model"))) +
  ggtitle(expression(paste("Recognition of ", beta, " values"))) +
  labs(colour=expression(beta))
