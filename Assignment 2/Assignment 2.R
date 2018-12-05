
# Assignment 2
# Nishit K Patel (nkp3@illinois.edu)

mypackages = c("leaps", "glmnet","gridExtra","gpglot2","reshape")   
tmp = setdiff(mypackages, rownames(installed.packages())) 
if (length(tmp) > 0) install.packages(tmp)

library(leaps)  # regsubsets
library(glmnet)  # glmnet for lasso
library(reshape)
library(ggplot2)
library(gridExtra)


set.seed(123)
# part 1
load("BostonHousing1.Rdata")
bh1 <- Housing1

n <- nrow(bh1)
p <- ncol(bh1) - 1

#convert in matrix form
X <- data.matrix(bh1[,-1])  
Y <- bh1[,1]  

err <- matrix(0,50,10)
vars <- matrix(0,50,10)
run_time <- matrix(0,50,10)
ridge.lambda <- matrix(0,50,4)
lasso.lambda <- matrix(0,50,4)


for(i in 1:50){
  
  # all.test.id: ntestxT matrix, each column records 
  ntest <- round(n * 0.25)  # test set size
  ntrain <- n-ntest  # training set size
  all.test.id <- matrix(0, ntest, 50)  # 
  for(t in 1:50){
    all.test.id[, t] <- sample(1:n, ntest)
  }
  
  save(all.test.id, file="alltestID.RData")
  
  test.id <- all.test.id[,i] 
  
  #full model
  start.time.full <- proc.time()[3]
  full.model <- lm(Y ~ ., data = bh1[-test.id,])
  Ytest.pred <- predict(full.model, newdata = bh1[test.id,])
  err[i,5] <- mean((Y[test.id] - Ytest.pred)^2)
  vars[i,5] <- length(full.model$coefficients) - 1
  run_time[i,5] <- proc.time()[3] - start.time.full
  
  
  #forward AIC
  start.time.aic_f <- proc.time()[3]
  full.model <- lm(Y ~ ., data = bh1[-test.id, ])
  stepAIC <- step(lm(Y ~ 1, data = bh1[-test.id, ]), 
                 list(upper = full.model),
                 trace = 0, direction = "forward")
  Ytest.pred <- predict(stepAIC, newdata = bh1[test.id, ])
  err[i,2] <- mean((Y[test.id] - Ytest.pred)^2)
  vars[i,2] <- length(stepAIC$coef) - 1  
  run_time[i,2] <- proc.time()[3] - start.time.aic_f
  
  
  #backward AIC
  start.time.aic_b <- proc.time()[3]
  full.model <- lm(Y ~ ., data = bh1[-test.id, ])
  stepAIC <- step(full.model, trace = 0, direction = "backward")
  Ytest.pred <- predict(stepAIC, newdata = bh1[test.id, ])
  err[i,1] <- mean((Y[test.id] - Ytest.pred)^2)
  vars[i,1] <- length(stepAIC$coef) - 1
  run_time[i,1] <- proc.time()[3] - start.time.aic_b
  
  
  #forward BIC
  start.time.bic_f <- proc.time()[3]
  full.model <- lm(Y ~ ., data = bh1[-test.id, ])
  stepAIC <- step(lm(Y ~ 1, data = bh1[-test.id, ]),
                 list(upper = full.model),
                 trace = 0, direction = "forward", k = log(ntrain))
  Ytest.pred <- predict(stepAIC, newdata = bh1[test.id, ])
  err[i,4] <- mean((Y[test.id] - Ytest.pred)^2)
  vars[i,4] <- length(stepAIC$coef) - 1 
  run_time[i,4] <- proc.time()[3] - start.time.bic_f
  
  
  #backward BIC
  start.time.bic_b <- proc.time()[3]
  full.model <- lm(Y ~ ., data = bh1[-test.id, ])
  stepAIC <- step(full.model, trace = 0,
                 direction = "backward", k = log(ntrain))
  Ytest.pred <- predict(stepAIC, newdata = bh1[test.id, ])
  err[i,3] <- mean((Y[test.id] - Ytest.pred)^2)
  vars[i,3] <- length(stepAIC$coef) - 1
  run_time[i,3] <- proc.time()[3] - start.time.bic_b
  
  
  #ridge min a
  start.time.rmin <- proc.time()[3]
  cv.out <- cv.glmnet(X[-test.id, ], Y[-test.id], alpha = 0)
  best.lam <- cv.out$lambda.min
  Ytest.pred <- predict(cv.out, s = best.lam, newx = X[test.id, ])
  err[i,10] <- mean((Y[test.id] - Ytest.pred)^2)
  run_time[,10] <- proc.time()[3] - start.time.rmin
  
  ntrain <-n - dim(all.test.id)[1]
  tmpX <- scale(X[-test.id, ]) * sqrt(ntrain / (ntrain - 1))
  d <- svd(tmpX)$d 
  vars[i,10] <- sum(d^2/(d^2 + best.lam*ntrain))
  
  #ridge 1se
  start.time.r1se <- proc.time()[3]
  cv.out <- cv.glmnet(X[-test.id, ], Y[-test.id], alpha = 0)
  lam.1se <- cv.out$lambda.1se
  Ytest.pred.1se <- predict(cv.out, s= lam.1se, newx = X[test.id,])
  err[i,9] <- mean((Y[test.id] - Ytest.pred.1se)^2)
  run_time[i,9] <- proc.time()[3] - start.time.r1se
  
  ntrain <-n - dim(all.test.id)[1]
  tmpX <- scale(X[-test.id, ]) * sqrt(ntrain / (ntrain - 1))
  d <- svd(tmpX)$d 
  vars[i,9] <- sum(d^2/(d^2 + lam.1se*ntrain))
  
  ridge.lambda[i,1] <- range(log(cv.out$lambda))[1]
  ridge.lambda[i,2] <- log(cv.out$lambda.min)
  ridge.lambda[i,3] <- log(cv.out$lambda.1se)
  ridge.lambda[i,4] <- range(log(cv.out$lambda))[2]
  
  #lasso min 
  start.time.lmin <- proc.time()[3]
  cv.out <- cv.glmnet(X[-test.id, ], Y[-test.id], alpha = 1)
  best.lam <- cv.out$lambda.min
  Ytest.pred <- predict(cv.out, s = best.lam, newx = X[test.id, ])
  err[i,7] <- mean((Ytest.pred - Y[test.id])^2)
  mylasso.coef <- predict(cv.out, s = best.lam, type = "coefficients")
  vars[i,7] <- sum(mylasso.coef != 0) - 1  # size of Lasso with lambda.min
  run_time[i,7] <- proc.time()[3] - start.time.lmin
  
  #lasso 1se 
  start.time.l1se <- proc.time()[3]
  cv.out <- cv.glmnet(X[-test.id, ], Y[-test.id], alpha = 1)
  best.lam <- cv.out$lambda.1se
  Ytest.pred <- predict(cv.out, s = best.lam, newx = X[test.id, ])
  err[i,6] <- mean((Ytest.pred - Y[test.id])^2)
  mylasso.coef <- predict(cv.out, s = best.lam, type = "coefficients")
  vars[i,6] <- sum(mylasso.coef != 0) - 1 # size of Lasso with lambda.1se
  run_time[i,6] <- proc.time()[3] - start.time.l1se
  
  lasso.lambda[i,1] <- range(log(cv.out$lambda))[1]
  lasso.lambda[i,2] <- log(cv.out$lambda.min)
  lasso.lambda[i,3] <- log(cv.out$lambda.1se)
  lasso.lambda[i,4] <- range(log(cv.out$lambda))[2]
  
  #refit using lasso 1se
  start.time.lrefit <- proc.time()[3]
  cv.out <- cv.glmnet(X[-test.id, ], Y[-test.id], alpha = 1)
  best.lam <- cv.out$lambda.1se
  Ytest.pred <- predict(cv.out, s = best.lam, newx = X[test.id, ])
  err[i,8] <- mean((Ytest.pred - Y[test.id])^2)
  mylasso.coef <- predict(cv.out, s = best.lam, type = "coefficients")
  
  var.sel <- row.names(mylasso.coef)[nonzeroCoef(mylasso.coef)[-1]]
  tmp.X <- X[, colnames(X) %in% var.sel]
  mylasso.refit <- coef(lm(Y[-test.id] ~ tmp.X[-test.id, ]))
  Ytest.pred <- mylasso.refit[1] + tmp.X[test.id,] %*% mylasso.refit[-1]
  err[i,8] <- mean((Ytest.pred - Y[test.id])^2)
  vars[i,8] <- sum(mylasso.coef != 0) - 1
  run_time[i,8] <- proc.time()[3] - start.time.lrefit
  
}

#lambda matrix
colnames(ridge.lambda) <- c("min_lambda","lambda.min","lambda.1se","max_lambda")
colnames(lasso.lambda) <- c("min_lambda","lambda.min","lambda.1se","max_lambda")
#ridge.lambda
#lasso.lambda


#refitting the ridge and lasso with lambda sequence
for(i in 1:50){
  
  # all.test.id: ntestxT matrix, each column records 
  ntest <- round(n * 0.25)  # test set size
  ntrain <- n-ntest  # training set size
  all.test.id <- matrix(0, ntest, 50)  # 
  for(t in 1:50){
    all.test.id[, t] <- sample(1:n, ntest)
  }
  
  save(all.test.id, file="alltestID.RData")
  
  test.id <- all.test.id[,i] 
  
  cv.out.time <- proc.time()[3]
  cv.out.r <- cv.glmnet(X[-test.id, ], Y[-test.id], alpha = 0,lambda = exp(seq(-4,-1,length=100)))
  Ytest.pred <- predict(cv.out.r, s = cv.out.r$lambda.min, newx = X[test.id, ])
  err[i,10] <- mean((Y[test.id] - Ytest.pred)^2)
  run_time[i,10] <- proc.time()[3] - cv.out.time
  
  ntrain <- n - dim(all.test.id)[1]
  tmpX <- scale(X[-test.id, ]) * sqrt(ntrain / (ntrain - 1))
  d <- svd(tmpX)$d 
  vars[i,10] <- sum(d^2/(d^2 + cv.out.r$lambda.min*ntrain))
 
  cv.out.time <- proc.time()[3]
  cv.out.r <- cv.glmnet(X[-test.id, ], Y[-test.id], alpha = 0,lambda =  exp(seq(-4,-1,length=100)))
  Ytest.pred <- predict(cv.out.r, s = cv.out.r$lambda.1se, newx = X[test.id, ])
  err[i,9] <- mean((Y[test.id] - Ytest.pred)^2)
  run_time[i,9] <- proc.time()[3] - cv.out.time
  
  ntrain <-n - dim(all.test.id)[1]
  tmpX <- scale(X[-test.id, ]) * sqrt(ntrain / (ntrain - 1))
  d <- svd(tmpX)$d 
  vars[i,9] <- sum(d^2/(d^2 + cv.out.r$lambda.1se*ntrain))
  
  cv.out.time <- proc.time()[3]
  cv.out.l <- cv.glmnet(X[-test.id, ], Y[-test.id], alpha = 1,lambda = exp(seq(-9,0,length=100)))
  Ytest.pred <- predict(cv.out.l, s = cv.out.l$lambda.min, newx = X[test.id, ])
  err[i,7] <- mean((Y[test.id] - Ytest.pred)^2)
  mylasso.coef <- predict(cv.out.l, s =cv.out.l$lambda.min, type = "coefficients")
  vars[i,7] <- sum(mylasso.coef != 0) - 1  # size of Lasso with lambda.min
  run_time[i,7] <- proc.time()[3] - cv.out.time
  
  cv.out.time <- proc.time()[3]
  cv.out.l <- cv.glmnet(X[-test.id, ], Y[-test.id], alpha = 1,lambda = exp(seq(-9,0,length=100)))
  Ytest.pred <- predict(cv.out.l, s = cv.out.l$lambda.1se, newx = X[test.id, ])
  err[i,6] <- mean((Y[test.id] - Ytest.pred)^2)
  mylasso.coef <- predict(cv.out.l, s = cv.out.l$lambda.1se, type = "coefficients")
  vars[i,6] <- sum(mylasso.coef != 0) - 1 # size of Lasso with lambda.1se
  run_time[i,6] <- proc.time()[3] - cv.out.time
  
  start.time.lrefit <- proc.time()[3]
  cv.out.l <- cv.glmnet(X[-test.id, ], Y[-test.id], alpha = 1,lambda = exp(seq(-9,0,length=100)))
  mylasso.coef <- predict(cv.out.l, s = cv.out.l$lambda.1se, type = "coefficients")
  
  var.sel <- row.names(mylasso.coef)[nonzeroCoef(mylasso.coef)[-1]]
  tmp.X <- X[, colnames(X) %in% var.sel]
  mylasso.refit <- coef(lm(Y[-test.id] ~ tmp.X[-test.id, ]))
  Ytest.pred <- mylasso.refit[1] + tmp.X[test.id,] %*% mylasso.refit[-1]
  err[i,8] <- mean((Ytest.pred - Y[test.id])^2)
  vars[i,8] <- sum(mylasso.coef != 0) - 1
  run_time[i,8] <- proc.time()[3] - start.time.lrefit
}


colnames(err) <- c("AIC.B","AIC.F","BIC.B","BIC.F","Full","L_1se","L_min","L_Refit","R_1se","R_min")
colnames(vars) <- c("AIC.B","AIC.F","BIC.B","BIC.F","Full","L_1se","L_min","L_Refit","R_1se","R_min")
colnames(run_time) <- c("AIC.B","AIC.F","BIC.B","BIC.F","Full","L_1se","L_min","L_Refit","R_1se","R_min")

#summary(err)
#summary(vars)

#error boxplot
err <- as.data.frame(err)
melterrData <- melt(err)
bh1_errplot <- ggplot(melterrData,aes(variable, value,color=variable)) +
  geom_boxplot() + 
  guides(fill=FALSE,color=FALSE)+ 
  xlab("Method") +
  ylab("Prediction Error")+ 
  labs(title="Boston Housingdata 1 error")

#model size box plot
vars <- as.data.frame(vars)
meltvarsData <- melt(vars)
bh1_msizeplot <- ggplot(meltvarsData,aes(variable, value,color=variable)) +
  geom_boxplot()+ 
  guides(fill=FALSE,color=FALSE)+ 
  xlab("Method") +
  ylab("Model Size")+ 
  labs(title="Boston Housingdata 1 model size")

#total run times
rt1 <- colSums(run_time,dims = 1)
rt1 <- as.data.frame(rt1)
rt1$names <- rownames(rt1)
meltrtData1 <- melt(rt1)
rt1_plot <- ggplot(meltrtData1,aes(x=names,y=value)) +
  geom_bar(stat = "identity")+ 
  geom_text(aes(label=round(value,2)), vjust=1.6, color="white", size=3.5) + 
  xlab("Method") +
  ylab("Run time")+ 
  labs(title="Boston Housingdata 3 total run time")


#part 2
load("BostonHousing2.Rdata")
bh2 <- Housing2

n2 <- nrow(bh2)
p2 <- ncol(bh2) - 1

#convert in matrix form
X2 <- data.matrix(bh2[,-1])  
Y2 <- bh2[,1]  

err2 <- matrix(0,50,5)
vars2 <- matrix(0,50,5)
run_time2 <- matrix(0,50,5)
ridge.lambda2 <- matrix(0,50,4)
lasso.lambda2 <- matrix(0,50,4)


for(i in 1:50){
  
  # all.test.id: ntestxT matrix, each column records 
  ntest <- round(n2 * 0.25)  # test set size
  ntrain <- n2-ntest  # training set size
  all.test.id <- matrix(0, ntest, 50)  # 
  for(t in 1:50){
    all.test.id[, t] <- sample(1:n2, ntest)
  }
  save(all.test.id, file="alltestID.RData")
  test.id <- all.test.id[,i] 
  
  
  #ridge min lambda
  start.time.rmin <- proc.time()[3]
  cv.out <- cv.glmnet(X2[-test.id, ], Y2[-test.id], alpha = 0)
  best.lam <- cv.out$lambda.min
  Ytest.pred <- predict(cv.out, s = best.lam, newx = X2[test.id, ])
  err2[i,5] <- mean((Y2[test.id] - Ytest.pred)^2)
  run_time2[i,5] <- proc.time()[3] - start.time.rmin
  
  ntrain <- n2 - dim(all.test.id)[1]
  tmpX <- scale(X2[-test.id, ]) * sqrt(ntrain / (ntrain - 1))
  d <- svd(tmpX)$d 
  vars2[i,5] <- sum(d^2/(d^2 + best.lam*ntrain))
  
  
  #ridge 1se lambda
  start.time.r1se <- proc.time()[3]
  cv.out <- cv.glmnet(X2[-test.id, ], Y2[-test.id], alpha = 0)
  lam.1se <- cv.out$lambda.1se
  Ytest.pred.1se <- predict(cv.out, s= lam.1se, newx = X2[test.id,])
  err2[i,4] <- mean((Y2[test.id] - Ytest.pred.1se)^2)
  run_time2[i,4] <- proc.time()[3] - start.time.r1se
  
  ntrain <- n2 - dim(all.test.id)[1]
  tmpX <- scale(X2[-test.id, ]) * sqrt(ntrain / (ntrain - 1))
  d <- svd(tmpX)$d 
  vars2[i,4] <- sum(d^2/(d^2 + lam.1se *ntrain))
  
  ridge.lambda2[i,1] <- range(log(cv.out$lambda))[1]
  ridge.lambda2[i,2] <- log(cv.out$lambda.min)
  ridge.lambda2[i,3] <- log(cv.out$lambda.1se)
  ridge.lambda2[i,4] <- range(log(cv.out$lambda))[2]
  
  #lasso min 
  start.time.lmin <- proc.time()[3]
  cv.out <- cv.glmnet(X2[-test.id, ], Y2[-test.id], alpha = 1)
  best.lam <- cv.out$lambda.min
  Ytest.pred <- predict(cv.out, s = best.lam, newx = X2[test.id, ])
  err2[i,2] <- mean((Ytest.pred - Y2[test.id])^2)
  mylasso.coef <- predict(cv.out, s = best.lam, type = "coefficients")
  vars2[i,2] <- sum(mylasso.coef != 0) - 1  # size of Lasso with lambda.min
  run_time2[i,2] <- proc.time()[3] - start.time.lmin
  
  #lasso 1se 
  start.time.l1se <- proc.time()[3]
  cv.out <- cv.glmnet(X2[-test.id, ], Y2[-test.id], alpha = 1)
  best.lam <- cv.out$lambda.1se
  Ytest.pred <- predict(cv.out, s = best.lam, newx = X2[test.id, ])
  err2[i,1] <- mean((Ytest.pred - Y2[test.id])^2)
  mylasso.coef <- predict(cv.out, s = best.lam, type = "coefficients")
  vars2[i,1] <- sum(mylasso.coef != 0) - 1 # size of Lasso with lambda.1se
  run_time2[i,1] <- proc.time()[3] - start.time.l1se
  
  lasso.lambda2[i,1] <- range(log(cv.out$lambda))[1]
  lasso.lambda2[i,2] <- log(cv.out$lambda.min)
  lasso.lambda2[i,3] <- log(cv.out$lambda.1se)
  lasso.lambda2[i,4] <- range(log(cv.out$lambda))[2]
  
  #refit using lasso 1se
  start.time.lrefit <- proc.time()[3]
  cv.out <- cv.glmnet(X2[-test.id, ], Y2[-test.id], alpha = 1)
  best.lam <- cv.out$lambda.1se
  mylasso.coef <- predict(cv.out, s = best.lam, type = "coefficients")
  
  var.sel <- row.names(mylasso.coef)[nonzeroCoef(mylasso.coef)[-1]]
  tmp.X2 <- X2[, colnames(X2) %in% var.sel]
  mylasso.refit <- coef(lm(Y2[-test.id] ~ tmp.X2[-test.id, ]))
  Ytest.pred <- mylasso.refit[1] + tmp.X2[test.id,] %*% mylasso.refit[-1]
  err2[i,3] <- mean((Ytest.pred - Y2[test.id])^2)
  mylasso.coef <- predict(cv.out, s = best.lam, type = "coefficients")
  vars2[i,3] <- sum(mylasso.refit != 0) - 1
  run_time2[i,3] <- proc.time()[3] - start.time.lrefit
  
}

#lambda matrix
colnames(ridge.lambda2) <- c("min_lambda","lambda.min","lambda.1se","max_lambda")
colnames(lasso.lambda2) <- c("min_lambda","lambda.min","lambda.1se","max_lambda")

#ridge.lambda2
#lasso.lambda2

#refitting the ridge and lasso with lambda sequence
for(i in 1:50){
  
  ntest <- round(n2 * 0.25)  # test set size
  ntrain <- n2-ntest  # training set size
  all.test.id <- matrix(0, ntest, 50)  # 
  for(t in 1:50){
    all.test.id[, t] <- sample(1:n2, ntest)
  }
  save(all.test.id, file="alltestID.RData")
  
  test.id <- all.test.id[,i] 
  
  #ridge min
  cv.out.time <- proc.time()[3]
  cv.out.r <- cv.glmnet(X2[-test.id, ], Y2[-test.id], alpha = 0,lambda = exp(seq(-4,0,length=100)))
  Ytest.pred <- predict(cv.out.r, s = cv.out.r$lambda.min, newx = X2[test.id, ])
  err2[i,5] <- mean((Y2[test.id] - Ytest.pred)^2)
  run_time2[i,5] <- proc.time()[3] - cv.out.time
  
  ntrain <- n2 - dim(all.test.id)[1]
  tmpX <- scale(X2[-test.id, ]) * sqrt(ntrain / (ntrain - 1))
  d <- svd(tmpX)$d 
  vars2[i,5] <- sum(d^2/(d^2 + cv.out.r$lambda.min*ntrain))
  
  #ridge 1se
  cv.out.time <- proc.time()[3]
  cv.out.r <- cv.glmnet(X2[-test.id, ], Y2[-test.id], alpha = 0,lambda =  exp(seq(-4,0,length=100)))
  Ytest.pred <- predict(cv.out.r, s = cv.out.r$lambda.1se, newx = X2[test.id, ])
  err2[i,4] <- mean((Y2[test.id] - Ytest.pred)^2)
  run_time2[i,4] <- proc.time()[3] - cv.out.time
  
  ntrain <- n2 - dim(all.test.id)[1]
  tmpX <- scale(X2[-test.id, ]) * sqrt(ntrain / (ntrain - 1))
  d <- svd(tmpX)$d 
  vars2[i,4] <- sum(d^2/(d^2 + cv.out.r$lambda.1se *ntrain))
  
  #lasso min
  cv.out.time <- proc.time()[3]
  cv.out.l <- cv.glmnet(X2[-test.id, ], Y2[-test.id], alpha = 1,lambda = exp(seq(-9,-5,length=100)))
  Ytest.pred <- predict(cv.out.l, s = cv.out.l$lambda.min, newx = X2[test.id, ])
  err2[i,2] <- mean((Y2[test.id] - Ytest.pred)^2)
  mylasso.coef <- predict(cv.out.l, s = cv.out.l$lambda.min, type = "coefficients")
  vars2[i,2] <- sum(mylasso.coef != 0) - 1  # size of Lasso with lambda.min
  run_time2[i,2] <- proc.time()[3] - cv.out.time
  
  #lasso 1se
  cv.out.time <- proc.time()[3]
  cv.out.l <- cv.glmnet(X2[-test.id, ], Y2[-test.id], alpha = 1,lambda = exp(seq(-9,-5,length=100)))
  Ytest.pred <- predict(cv.out.l, s = cv.out.l$lambda.1se, newx = X2[test.id, ])
  err2[i,1] <- mean((Y2[test.id] - Ytest.pred)^2)
  mylasso.coef <- predict(cv.out.l, s = cv.out.l$lambda.1se, type = "coefficients")
  vars2[i,1] <- sum(mylasso.coef != 0) - 1 # size of Lasso with lambda.1se
  run_time2[i,1] <- proc.time()[3] - cv.out.time
  
  #refit using lasso 1se
  start.time.lrefit <- proc.time()[3]
  cv.out.l <- cv.glmnet(X2[-test.id, ], Y2[-test.id], alpha = 1,lambda = exp(seq(-9,-5,length=100)))
  mylasso.coef <- predict(cv.out.l, s = cv.out.l$lambda.1se, type = "coefficients")
  
  var.sel <- row.names(mylasso.coef)[nonzeroCoef(mylasso.coef)[-1]]
  tmp.X2 <- X2[, colnames(X2) %in% var.sel]
  mylasso.refit <- coef(lm(Y2[-test.id] ~ tmp.X2[-test.id, ]))
  Ytest.pred <- mylasso.refit[1] + tmp.X2[test.id,] %*% mylasso.refit[-1]
  err2[i,3] <- mean((Ytest.pred - Y2[test.id])^2)
  mylasso.coef <- predict(cv.out.l, s = cv.out.l$lambda.1se, type = "coefficients")
  vars2[i,3] <- sum(mylasso.refit != 0) - 1
  run_time2[i,3] <- proc.time()[3] - start.time.lrefit
}

colnames(err2) <- c("L_1se","L_min","L_Refit","R_1se","R_min")
#summary(err2)

#error plot
err2 <- as.data.frame(err2)
melterrData2 <- melt(err2)
bh2_errplot <- ggplot(melterrData2,aes(variable, value,color=variable)) +
  geom_boxplot()+
  guides(fill=FALSE,color=FALSE)+ 
  xlab("Method") +
  ylab("Predicton Error")+ 
  labs(title="Boston Housingdata 2 error")

colnames(vars2) <- c("L_1se","L_min","L_Refit","R_1se","R_min")
#summary(vars2)
#model size box plot
vars2 <- as.data.frame(vars2)
meltvarsData2 <- melt(vars2)
bh2_msizeplot <- ggplot(meltvarsData2,aes(variable, value,color=variable)) +
  geom_boxplot()+
  guides(fill=FALSE,color=FALSE)+ 
  xlab("Method") +
  ylab("Model Size")+ 
  labs(title="Boston Housingdata 2 model size")

#runtime
colnames(run_time2) <- c("L_1se","L_min","L_Refit","R_1se","R_min")
rt2 <- colSums(run_time2, dims = 1)
rt2 <- as.data.frame(rt2)
rt2$names <- rownames(rt2)
meltrtData2 <- melt(rt2)
rt2_plot <- ggplot(meltrtData2,aes(x=names, y=value)) +
  geom_bar(stat = "identity")+ 
  geom_text(aes(label=round(value,2)), vjust=1.6, color="white", size=3.5) + 
  xlab("Method") +
  ylab("Run time")+ 
  labs(title="Boston Housingdata 2 total run time") 


#part 3
load("BostonHousing3.Rdata")
bh3 <- Housing3

n3 <- nrow(bh3)
p3 <- ncol(bh3) - 1

#convert in matrix form
X3 <- data.matrix(bh3[,-1])  
Y3 <- bh3[,1]  

err3 <- matrix(0,50,5)
vars3 <- matrix(0,50,5)
run_time3 <- matrix(0,50,5)
ridge.lambda3 <- matrix(0,50,4)
lasso.lambda3 <- matrix(0,50,4)


for(i in 1:50){
  
  # all.test.id: ntestxT matrix, each column records 
  ntest <- round(n3 * 0.25)  # test set size
  ntrain <- n3-ntest  # training set size
  all.test.id <- matrix(0, ntest, 50)  # 
  for(t in 1:50){
    all.test.id[, t] <- sample(1:n3, ntest)
  }
  save(all.test.id, file="alltestID.RData")
  test.id <- all.test.id[,i] 
  
  
  #ridge min a
  start.time.rmin <- proc.time()[3]
  cv.out <- cv.glmnet(X3[-test.id, ], Y3[-test.id], alpha = 0)
  best.lam <- cv.out$lambda.min
  Ytest.pred <- predict(cv.out, s = best.lam, newx = X3[test.id, ])
  err3[i,5] <- mean((Y3[test.id] - Ytest.pred)^2)
  run_time3[i,5] <- proc.time()[3] - start.time.rmin
  
  ntrain <- n3 - dim(all.test.id)[1]
  tmpX <- scale(X3[-test.id, ]) * sqrt(ntrain / (ntrain - 1))
  d <- svd(tmpX)$d 
  vars3[i,5] <- sum(d^2/(d^2 + best.lam*ntrain))
  
  
  #ridge 1se
  start.time.r1se <- proc.time()[3]
  cv.out <- cv.glmnet(X3[-test.id, ], Y3[-test.id], alpha = 0)
  lam.1se <- cv.out$lambda.1se
  Ytest.pred.1se <- predict(cv.out, s= lam.1se, newx = X3[test.id,])
  err3[i,4] <- mean((Y3[test.id] - Ytest.pred.1se)^2)
  run_time3[i,4] <- proc.time()[3] - start.time.r1se
  
  ntrain <- n3 - dim(all.test.id)[1]
  tmpX <- scale(X3[-test.id, ]) * sqrt(ntrain / (ntrain - 1))
  d <- svd(tmpX)$d 
  vars3[i,4] <- sum(d^2/(d^2 + lam.1se *ntrain))
  
  
  ridge.lambda3[i,1] <- range(log(cv.out$lambda))[1]
  ridge.lambda3[i,2] <- log(cv.out$lambda.min)
  ridge.lambda3[i,3] <- log(cv.out$lambda.1se)
  ridge.lambda3[i,4] <- range(log(cv.out$lambda))[2]
  
  #lasso min 
  start.time.lmin <- proc.time()[3]
  cv.out <- cv.glmnet(X3[-test.id, ], Y3[-test.id], alpha = 1)
  best.lam <- cv.out$lambda.min
  Ytest.pred <- predict(cv.out, s = best.lam, newx = X3[test.id, ])
  err3[i,2] <- mean((Ytest.pred - Y3[test.id])^2)
  mylasso.coef <- predict(cv.out, s = best.lam, type = "coefficients")
  vars3[i,2] <- sum(mylasso.coef != 0) - 1  # size of Lasso with lambda.min
  run_time3[i,2] <- proc.time()[3] - start.time.lmin
  
  #lasso 1se 
  start.time.l1se <- proc.time()[3]
  cv.out <- cv.glmnet(X3[-test.id, ], Y3[-test.id], alpha = 1)
  best.lam <- cv.out$lambda.1se
  Ytest.pred <- predict(cv.out, s = best.lam, newx = X3[test.id, ])
  err3[i,1] <- mean((Ytest.pred - Y3[test.id])^2)
  mylasso.coef <- predict(cv.out, s = best.lam, type = "coefficients")
  vars3[i,1] <- sum(mylasso.coef != 0) - 1 # size of Lasso with lambda.1se
  run_time3[i,1] <- proc.time()[1] - start.time.l1se
  
  lasso.lambda3[i,1] <- range(log(cv.out$lambda))[1]
  lasso.lambda3[i,2] <- log(cv.out$lambda.min)
  lasso.lambda3[i,3] <- log(cv.out$lambda.1se)
  lasso.lambda3[i,4] <- range(log(cv.out$lambda))[2]
  
  #refit using lasso 1se
  start.time.lrefit <- proc.time()[1]
  var.sel <- row.names(mylasso.coef)[nonzeroCoef(mylasso.coef)[-1]]
  tmp.X3 <- X3[, colnames(X3) %in% var.sel]
  mylasso.refit <- coef(lm(Y3[-test.id] ~ tmp.X3[-test.id, ]))
  Ytest.pred <- mylasso.refit[1] + tmp.X3[test.id,] %*% mylasso.refit[-1]
  err3[i,3] <- mean((Ytest.pred - Y3[test.id])^2)
  mylasso.coef <- predict(cv.out, s = best.lam, type = "coefficients")
  vars3[i,3] <- sum(mylasso.refit != 0) - 1
  run_time3[i,3] <- proc.time()[3] - start.time.lrefit
}

#lambda matrix
colnames(ridge.lambda3) <- c("min_lambda","lambda.min","lambda.1se","max_lambda")
colnames(lasso.lambda3) <- c("min_lambda","lambda.min","lambda.1se","max_lambda")

#ridge.lambda3
#lasso.lambda3


for(i in 1:50){
  
  # all.test.id: ntestxT matrix, each column records 
  ntest <- round(n3 * 0.25)  # test set size
  ntrain <- n3-ntest  # training set size
  all.test.id <- matrix(0, ntest, 50)  # 
  for(t in 1:50){
    all.test.id[, t] <- sample(1:n3, ntest)
  }
  save(all.test.id, file="alltestID.RData")
  test.id <- all.test.id[,i] 
  
  
  #ridge min a
  start.time.rmin <- proc.time()[3]
  cv.out.r <- cv.glmnet(X3[-test.id, ], Y3[-test.id], alpha = 0, lambda = exp(seq(1,2,length=100)))
  Ytest.pred <- predict(cv.out.r, s = cv.out.r$lambda.min, newx = X3[test.id, ])
  err3[i,5] <- mean((Y3[test.id] - Ytest.pred)^2)
  run_time3[,5] <- proc.time()[3] - start.time.rmin
  
  ntrain <- n3 - dim(all.test.id)[1]
  tmpX <- scale(X3[-test.id, ]) * sqrt(ntrain / (ntrain - 1))
  d <- svd(tmpX)$d 
  vars3[i,5] <- sum(d^2/(d^2 + cv.out.r$lambda.min*ntrain))
  
  
  #ridge 1se
  start.time.r1se <- proc.time()[3]
  cv.out.r <- cv.glmnet(X3[-test.id, ], Y3[-test.id], alpha = 0, lambda = exp(seq(1,2,length=100)))
  Ytest.pred.1se <- predict(cv.out.r, s= cv.out.r$lambda.1se, newx = X3[test.id,])
  err3[i,4] <- mean((Y3[test.id] - Ytest.pred.1se)^2)
  run_time3[i,4] <- proc.time()[3] - start.time.r1se
  
  ntrain <- n3 - dim(all.test.id)[1]
  tmpX <- scale(X3[-test.id, ]) * sqrt(ntrain / (ntrain - 1))
  d <- svd(tmpX)$d 
  vars3[i,4] <- sum(d^2/(d^2 + cv.out.r$lambda.1se *ntrain))
  
  
  #lasso min 
  start.time.lmin <- proc.time()[3]
  cv.out.l <- cv.glmnet(X3[-test.id, ], Y3[-test.id], alpha = 1, lambda = exp(seq(-6,-2,length=100)))
  Ytest.pred <- predict(cv.out.l, s = cv.out.l$lambda.min, newx = X3[test.id, ])
  err3[i,2] <- mean((Ytest.pred - Y3[test.id])^2)
  mylasso.coef <- predict(cv.out.l, s = cv.out.l$lambda.min, type = "coefficients")
  vars3[i,2] <- sum(mylasso.coef != 0) - 1  # size of Lasso with lambda.min
  run_time3[i,2] <- proc.time()[3] - start.time.lmin
  
  #lasso 1se 
  start.time.l1se <- proc.time()[3]
  cv.out.l <- cv.glmnet(X3[-test.id, ], Y3[-test.id], alpha = 1, lambda = exp(seq(-6,-2,length=100)))
  Ytest.pred <- predict(cv.out.l, s = cv.out.l$lambda.1se, newx = X3[test.id, ])
  err3[i,1] <- mean((Ytest.pred - Y3[test.id])^2)
  mylasso.coef <- predict(cv.out.l, s = cv.out.l$lambda.1se, type = "coefficients")
  vars3[i,1] <- sum(mylasso.coef != 0) - 1 # size of Lasso with lambda.1se
  run_time3[i,1] <- proc.time()[3] - start.time.l1se
  
  
  #refit using lasso 1se
  start.time.lrefit <- proc.time()[3]
  cv.out.l <- cv.glmnet(X3[-test.id, ], Y3[-test.id], alpha = 1, lambda = exp(seq(-6,-2,length=100)))
  mylasso.coef <- predict(cv.out.l, s = cv.out.l$lambda.1se, type = "coefficients")
  
  var.sel <- row.names(mylasso.coef)[nonzeroCoef(mylasso.coef)[-1]]
  tmp.X3 <- X3[, colnames(X3) %in% var.sel]
  mylasso.refit <- coef(lm(Y3[-test.id] ~ tmp.X3[-test.id, ]))
  Ytest.pred <- mylasso.refit[1] + tmp.X3[test.id,] %*% mylasso.refit[-1]
  err3[i,3] <- mean((Ytest.pred - Y3[test.id])^2)
  mylasso.coef <- predict(cv.out.l, s = cv.out.l$lambda.1se, type = "coefficients")
  vars3[i,3] <- sum(mylasso.refit != 0) - 1
  run_time3[i,3] <- proc.time()[3] - start.time.lrefit
}

colnames(err3) <- c("L_1se","L_min","L_Refit","R_1se","R_min")
#summary(err3)

#error plot
err3 <- as.data.frame(err3)
melterrData3 <- melt(err3)
bh3_errplot <- ggplot(melterrData3,aes(variable, value,color=variable)) +
  geom_boxplot()+ 
  guides(fill=FALSE,color=FALSE)+ 
  xlab("Method") +
  ylab("Prediction Error")+ 
  labs(title="Boston Housingdata 3 error")

colnames(vars3) <- c("L_1se","L_min","L_Refit","R_1se","R_min")
#summary(vars3)

#model size box plot
vars3 <- as.data.frame(vars3)
meltvarsData3 <- melt(vars3)
bh3_msizeplot <- ggplot(meltvarsData3,aes(variable, value,color=variable)) +
  geom_boxplot()+ 
  guides(fill=FALSE,color=FALSE)+ 
  xlab("Method") +
  ylab("Model Size")+ 
  labs(title="Boston Housingdata 3 model size")
 

#run time
colnames(run_time3) <- c("L_1se","L_min","L_Refit","R_1se","R_min")
rt3 <- colSums(run_time3, dims = 1)
rt3 <- as.data.frame(rt3)
rt3$names <- rownames(rt3)
meltrtData3 <- melt(rt3)
rt3_plot <- ggplot(meltrtData3,aes(x=names,y=value)) +
  geom_bar(stat = "identity")+ 
  geom_text(aes(label=round(value,2)), vjust=1.6, color="white", size=3.5) + 
  xlab("Method") +
  ylab("Run time")+ 
  labs(title="Boston Housingdata 3 total run time")



pdf("output.pdf", width=15,height=6)
grid.arrange(bh1_errplot,bh1_msizeplot,nrow=2)
grid.arrange(bh2_errplot,bh2_msizeplot,bh3_errplot,bh3_msizeplot,nrow=2,ncol=2)
grid.arrange(rt1_plot,rt2_plot,rt3_plot,nrow=3)
dev.off()



