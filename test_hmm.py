import numpy as np

from hmm import HMM, read_hmm, read_sequence


hmmfile = "test.hmm"
seqfile = "test.seq"

M, N, pi, A, B = read_hmm(hmmfile)
T, obs = read_sequence(seqfile)
	
hmm_object = HMM(pi, A, B)

#test forward algorithm
prob, alpha = hmm_object.forward(obs)
print "forward probability is %f" % np.log(prob)
prob, alpha, scale = hmm_object.forward_with_scale(obs)
print "forward probability with scale is %f" % prob

# test backward algorithm
prob, beta = hmm_object.backward(obs)
print "backward probability is %f" % prob 
beta = hmm_object.backward_with_scale(obs, scale)

# test baum-welch algorithm
logprobinit, logprobfinal = hmm_object.baum_welch(obs)
print "------------------------------------------------"
print "estimated parameters are: "
print "pi is:"
print hmm_object.pi
print "A is:"
print hmm_object.A
print "B is:"
print hmm_object.B
print "------------------------------------------------"
print "initial log probability is:"
print logprobinit
print "final log probability is:"
print logprobfinal