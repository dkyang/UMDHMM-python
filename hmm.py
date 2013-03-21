import numpy as np

DELTA = 0.001

class HMM:


    def __init__(self, pi, A, B):
        self.pi = pi
        self.A = A
        self.B = B
        self.M = B.shape[1]
        self.N = A.shape[0]
        
    def forward(self,obs):
        T = len(obs)
        N = self.N
		
        alpha = np.zeros([N,T])
        alpha[:,0] = self.pi[:] * self.B[:,obs[0]-1]                                                                                                      
	
        for t in xrange(1,T):
            for n in xrange(0,N):
                alpha[n,t] = np.sum(alpha[:,t-1] * self.A[:,n]) * self.B[n,obs[t]-1]
    		 		
        prob = np.sum(alpha[:,T-1])
        return prob, alpha
		
    def forward_with_scale(self, obs):
        """see scaling chapter in "A tutorial on hidden Markov models and 
	    selected applications in speech recognition." 
	    """
        T = len(obs)
        N = self.N
        alpha = np.zeros([N,T])
        scale = np.zeros(T)

        alpha[:,0] = self.pi[:] * self.B[:,obs[0]-1]
        scale[0] = np.sum(alpha[:,0])
        alpha[:,0] /= scale[0]

        for t in xrange(1,T):
            for n in xrange(0,N):
                alpha[n,t] = np.sum(alpha[:,t-1] * self.A[:,n]) * self.B[n,obs[t]-1]
            scale[t] = np.sum(alpha[:,t])
            alpha[:,t] /= scale[t]

        logprob = np.sum(np.log(scale[:]))
        return logprob, alpha, scale	
		
    def backward(self, obs):
        T = len(obs)
        N = self.N
        beta = np.zeros([N,T])
        
        beta[:,T-1] = 1
        for t in reversed(xrange(0,T-1)):
            for n in xrange(0,N):
			    beta[n,t] = np.sum(self.B[:,obs[t+1]-1] * self.A[n,:] * beta[:,t+1])
				
        prob = np.sum(beta[:,0])
        return prob, beta

    def backward_with_scale(self, obs, scale):
        T = len(obs)
        N = self.N
        beta = np.zeros([N,T])

        beta[:,T-1] = 1 / scale[T-1]
        for t in reversed(xrange(0,T-1)):
            for n in xrange(0,N):
                beta[n,t] = np.sum(self.B[:,obs[t+1]-1] * self.A[n,:] * beta[:,t+1])
                beta[n,t] /= scale[t]
		
        return beta

    def viterbi(self, obs):
        T = len(obs)
        N = self.N
        psi = np.zeros([N,T]) # reverse pointer
        delta = np.zeros([N,T])
        q = np.zeros(T)
        temp = np.zeros(N)        
		
        delta[:,0] = self.pi[:] * self.B[:,obs[0]-1]	
		
        for t in xrange(1,T):
            for n in xrange(0,N):
                temp = delta[:,t-1] * self.A[:,n]	
                max_ind = argmax(temp)
                psi[n,t] = max_ind
                delta[n,t] = self.B[n,obs[t]-1] * temp[max_ind]

        max_ind = argmax(delta[:,T-1])
        q[T-1] = max_ind
        prob = delta[:,T-1][max_ind]

        for t in reversed(xrange(1,T-1)):
            q[t] = psi[q[t+1],t+1]	
			
        return prob, q, delta	
		
    def viterbi_log(self, obs):
        
        T = len(obs)
        N = self.N
        psi = np.zeros([N,T])
        delta = np.zeros([N,T])
        pi = np.zeros(self.pi.shape)
        A = np.zeros(self.A.shape)
        biot = np.zeros([N,T])

        pi = np.log(self.pi)		
        A = np.log(self.A)
        biot = np.log(self.B[:,obs[:]-1])

        delta[:,0] = pi[:] + biot[:,0]

        for t in xrange(1,T):
            for n in xrange(0,N):
                temp = delta[:,t-1] + self.A[:,n]	
                max_ind = argmax(temp)
                psi[n,t] = max_ind
                delta[n,t] = temp[max_ind] + biot[n,t]   

        max_ind = argmax(delta[:,T-1])
        q[T-1] = max_ind			
        logprob = delta[max_ind,T-1]  	
        		
        for t in reversed(xrange(1,T-1)):
            q[t] = psi[q[t+1],t+1]	

        return logprob, q, delta

    def baum_welch(self, obs):
        T = len(obs)
        M = self.M
        N = self.N		
        alpha = np.zeros([N,T])
        beta = np.zeros([N,T])
        scale = np.zeros(T)
        gamma = np.zeros([N,T])
        xi = np.zeros([N,N,T-1])
    
        # caculate initial parameters
        logprobprev, alpha, scale = self.forward_with_scale(obs)
        beta = self.backward_with_scale(obs, scale)			
        gamma = self.compute_gamma(alpha, beta)	
        xi = self.compute_xi(obs, alpha, beta)	
        logprobinit = logprobprev		
		
        # start interative 
        while True:
		    # E-step
            self.pi = 0.001 + 0.999*gamma[:,0]
            for i in xrange(N):
                denominator = np.sum(gamma[i,0:T-1])
                for j in xrange(N): 
                    numerator = np.sum(xi[i,j,0:T-1])
                    self.A[i,j] = numerator / denominator
                   				
            self.A = 0.001 + 0.999*self.A
            for j in xrange(0,N):
                denominator = np.sum(gamma[j,:])
                for k in xrange(0,M):
                    numerator = 0.0
                    for t in xrange(0,T):
                        if obs[t]-1 == k:
                            numerator += gamma[j,t]
                    self.B[j,k] = numerator / denominator
            self.B = 0.001 + 0.999*self.B

            # M-step
            logprobcur, alpha, scale = self.forward_with_scale(obs)
            beta = self.backward_with_scale(obs, scale)			
            gamma = self.compute_gamma(alpha, beta)	
            xi = self.compute_xi(obs, alpha, beta)	

            delta = logprobcur - logprobprev
            logprobprev = logprobcur
            # print "delta is ", delta
            if delta <= DELTA:
                break 	
				
        logprobfinal = logprobcur
        return logprobinit, logprobfinal				
			
    def compute_gamma(self, alpha, beta):
        gamma = np.zeros(alpha.shape)
        gamma = alpha[:,:] * beta[:,:]
        gamma = gamma / np.sum(gamma,0)
        return gamma
			
    def compute_xi(self, obs, alpha, beta):
        T = len(obs)
        N = self.N
        xi = np.zeros((N, N, T-1))
			
        for t in xrange(0,T-1):        
            for i in xrange(0,N):
                for j in xrange(0,N):
                    xi[i,j,t] = alpha[i,t] * self.A[i,j] * \
                                self.B[j,obs[t+1]-1] * beta[j,t+1]
            xi[:,:,t] /= np.sum(np.sum(xi[:,:,t],1),0)	
        return xi

def read_hmm(hmmfile):
    fhmm = open(hmmfile,'r') 

    M = int(fhmm.readline().split(' ')[1])
    N = int(fhmm.readline().split(' ')[1]) 
	
    A = np.array([])
    fhmm.readline()
    for i in xrange(N):
        line = fhmm.readline()
        if i == 0:
            A = np.array(map(float,line.split(',')))
        else:
            A = np.vstack((A,map(float,line.split(','))))
		
    B = np.array([])
    fhmm.readline()
    for i in xrange(N):
        line = fhmm.readline()
        if i == 0:
            B = np.array(map(float,line.split(',')))
        else:
            B = np.vstack((B,map(float,line.split(','))))
    
    fhmm.readline()
    line = fhmm.readline()
    pi = np.array(map(float,line.split(',')))
	
    fhmm.close()
    return M, N, pi, A, B 
	
def read_sequence(seqfile):
    fseq = open(seqfile,'r') 
	
    T = int(fseq.readline().split(' ')[1])
    line = fseq.readline()
    obs = np.array(map(int,line.split(',')))
	
    fseq.close()
    return T, obs
    	
