

# Assignment 3 Bonus credit
# Nishit K Patel (nkp3@illinois.edu)


mypackages = c("gridExtra")   
tmp = setdiff(mypackages, rownames(installed.packages())) 
if (length(tmp) > 0) install.packages(tmp)

library("gridExtra")

lo.lev <- function(x1, sp){
  
  ## YOUR CODE: compute the diagonal entries of the smoother
  ##             matrix S, stored in vector "lev"
  ## Tip: check how we compute the smoother matrix
  ##      for smoothing spline models
  
  len <- length(x1)
  dgnl <- diag(len)
  lev <- rep(0,length=len)
  
  for(j in 1:len){
    y <- dgnl[,j]
    loe.fit <- loess(y~x1, span = sp,control = loess.control(surface = "direct"))
    y_hat <- fitted.values(loe.fit)
    lev[j] <- y_hat[j]
  }
  return(lev)
}

onestep_CV <- function(x1, y1, sp){
  
  ## YOUR CODE: 
  ## 1) fit a loess model y1 ~ x1 with span = sp, and extract 
  ##    the corresponding residual vector
  ## 2) call lo.lev to obtain the diagonal entries of S
  ## 3) compute LOO-CV and GCV using formula from lecture notes
  ##    [lec_W5_NonlinearRegression.pdf] page 33. 
  
  assign("data", data.frame(x=x1,y=y1))
  len <- length(x1)
  loe.fit <- loess(y ~ x, data=data,span = sp,control = loess.control(surface = "direct"))
  res <- residuals(loe.fit)
  
  s_diag <- lo.lev(x1,sp)
  tr_hat <- sum(s_diag)
  sse <- sum(res^2)
  
  cv <- sum((res/(1 - s_diag))^2)/len
  gcv <- sse/(len * (1 - (tr_hat/len))^2)
  
  return(list(cv = cv, gcv = gcv))
}

myCV <- function(x1, y1, span){
  ## x1, y1: two vectors
  ## span: a sequence of values for "span"
  
  m = length(span)
  cv = rep(0, m)
  gcv = rep(0, m)
  for(i in 1:m){
    tmp = onestep_CV(x1, y1, span[i])
    cv[i] = tmp$cv
    gcv[i] = tmp$gcv
  }
  return(list(cv = cv, gcv = gcv))
}


#read data
mydata = read.csv(file = "Coding3_Bonus_Data.csv")

span1 = seq(from = 0.2, by = 0.05, length = 15)
cv.out = myCV(mydata$x, mydata$y, span1)
cv = cv.out$cv
gcv = cv.out$gcv
df <- cbind(cv, gcv)
colnames(df) <- c("CV","GCV")

#create pdf
pdf("AssignmentOutput_3_Bonus_4007_nkp3_NishitPatel.pdf", width=15,height=6)
grid.table(df)

span1[gcv == min(gcv)]
span1[cv == min(cv)]

par(mfrow=c(1,2))
plot(span1, gcv, type = "n", xlab = "span", ylab = "GCV")
points(span1, gcv, pch = 3, col = "blue")
plot(span1, cv, type = "n", xlab = "span", ylab = "CV")
points(span1, cv, pch = 3, col = "blue")
dev.off()

