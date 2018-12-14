
#assignment 3
# Nishit K Patel (nkp3@illinois.edu)

mypackages = c("gridExtra")   
tmp = setdiff(mypackages, rownames(installed.packages())) 
if (length(tmp) > 0) install.packages(tmp)

library("gridExtra")

myBW = function(x, A, B, w, n.iter = 100){
  # Input: 
  # x: T-by-1 observation sequence
  # A: initial estimate for mz-by-mz transition matrix
  # B: initial estimate for mz-by-mx emission matrix
  # w: initial estimate for mz-by-1 initial distribution over Z_1
  # Output MLE of A and B; we do not update w
  # list(A = A, B=B, w = w)
  
  for(i in 1:n.iter){
    update.para = BW.onestep(x, A, B, w)
    A = update.para$A
    B = update.para$B
  }
  return(list(A = A, B = B, w = w)) 
}


BW.onestep = function(x, A, B, w){
  # Input: 
  # x: T-by-1 observation sequence
  # A: current estimate for mz-by-mz transition matrix
  # B: current estimate for mz-by-mx emission matrix
  # w: current estimate for mz-by-1 initial distribution over Z_1
  # Output the updated parameters 
  # para = list(A = A1, B = B1)
  
  # We DO NOT update the initial distribution w
  
  T = length(x)
  mz = nrow(A)
  alp = forward.prob(x, A, B, w)
  beta = backward.prob(x, A, B, w)
  myGamma = array(0, dim=c(mz, mz, T-1))
  
  ###
  ## YOUR CODE: 
  ## Compute gamma_t(i,j), which are stored in myGamma
  ##
  #500 matrix of 2x2
  # for(t in 1:T-1){
  #   myGamma[,,t] <- (matrix(alp[t,]) %*% t(A %*% matrix(beta[t+1,]* B[,x[t+1]])))   
  # }
  
  for(t in 1:(T-1)){
    for(i in 1:mz){
      for (j in 1:mz){
        myGamma[i,j,t] = alp[t,i] * A[i,j] * B[j,x[t+1]] * beta[t+1,j]
      }
    }
  }
  
  
  A = rowSums(myGamma, dims = 2)
  A = A/rowSums(A)
  
  tmp = apply(myGamma, c(1, 3), sum)  # mz-by-(T-1)
  tmp = cbind(tmp, colSums(myGamma[, , T-1]))
  for(l in 1:mx){
    B[, l] = rowSums(tmp[, which(x==l)])
  }
  B = B/rowSums(B)
  return(list(A = A, B = B))
}


forward.prob = function(x, A, B, w){
  
  # Output the forward probability matrix alp 
  # alp: T by mz, (t, i) entry = P(x_{1:t}, Z_t = i)
  
  T = length(x)
  mz = nrow(A)
  alp = matrix(0, T, mz)
  
  # fill in the first row of alp
  alp[1, ] = w*B[, x[1]]
  
  # Recursively compute the remaining rows of alp
  for(t in 2:T){
    tmp = alp[t-1, ] %*% A
    alp[t, ] = tmp * B[, x[t]]
  }
  return(alp)
}

backward.prob = function(x, A, B, w){
  # Output the backward probability matrix beta
  # beta: T by mz, (t, i) entry = P(x_{(t+1):n} | Z_t = i)
  # for t=1, ..., n-1
  
  T = length(x)
  mz = nrow(A)
  beta = matrix(1, T, mz)
  
  # The last row of beta is all 1.
  # Recursively compute the previous rows of beta
  for(t in (T-1):1){
    tmp = as.matrix(beta[t+1, ] * B[, x[t+1]])  # make tmp a column vector
    beta[t, ] = t(A %*% tmp)
  }
  return(beta)
}


myViterbi <- function(obs, transition_mat, emission_mat, initial_mat){
  
  T <- length(obs)
  A <- transition_mat
  
  pi <- initial_mat
  B <- emission_mat
  
  #Initialization:
  delta <- matrix(data=NA, nrow=2, ncol=length(data$X))
  delta[1,1] <- log(pi[1]*B[1,data$X[1]])
  delta[2,1] <- log(pi[2]*B[2,data$X[1]])
  
  pis <- matrix(data=NA, nrow=2, ncol=length(data$X))
  
  
  for(t in 2:T) {
    delta[1,t] <- max(c(delta[1,t-1] + log(A[1,1]), delta[2,t-1] + log(A[2,1]))) + log(B[1,data$X[t]])
    delta[2,t] <- max(c(delta[1,t-1] + log(A[1,2]), delta[2,t-1] + log(A[2,2]))) + log(B[2,data$X[t]])
    
    pis[1,t] <- ifelse(delta[1,t-1] + log(A[1,1]) > delta[2,t-1] + log(A[2,1]), yes=1, no=2)
    pis[2,t] <- ifelse(delta[1,t-1] + log(A[1,2]) > delta[2,t-1] + log(A[2,2]), yes=1, no=2)
  }
  
  
  #Termination:
  prob_Sequence <- NULL
  prob_Sequence[T] <- ifelse(delta[1,T] > delta[2,T], yes=1, no=2)

  #Backtracking:
  for(t in (T-1):1) {
    prob_Sequence[t] <- pis[prob_Sequence[t+1],t+1]
  }
  prob_Sequence <- ifelse(prob_Sequence == 1, "A", "B")
  prob_Sequence
}

  
data = read.csv("Coding3_HMM_Data.csv")

mz=2; mx=3
ini.A = matrix(1, mz, mz)
ini.A = ini.A/rowSums(ini.A)
ini.B = matrix(1:6, mz, mx)
ini.B = ini.B/rowSums(ini.B)
ini.w = c(1/2, 1/2)

myout = myBW(data$X, ini.A, ini.B, ini.w, n.iter = 100)
myout.Z = myViterbi(data$X, myout$A, myout$B, ini.w)
write.table(myout.Z, file = "Coding3_HMM_Viterbi_Output.txt", 
            row.names = FALSE, col.names = FALSE)


df1 <- tableGrob(myout$A, rows = c("1","2"), cols = c("1","2"))
df2 <- tableGrob(myout$B, rows= c("1","2"), cols = c("1","2","3"))

# create pdf
pdf("AssignmentOutput_3_HMM.pdf", width=15,height=6)
grid.arrange(df1,df2, ncol = 1, nrow = 2)
dev.off()

