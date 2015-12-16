#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Yin Yang Latent Dirichlet Allocation using Collapsed Gibbs Sampling
# This code is available under the MIT License.
# (c)2015 Daniel Rugeles 
# daniel007[at]e.ntu.edu.sg / Nanyang Technological University



if __name__ == "__main__":

	import numpy
	import optparse
	import csv
	import ylda

	#**1. Parse Settings  
	parser = optparse.OptionParser()
	parser.add_option("-f", dest="filename", help="filename of the dataset")
	parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
	parser.add_option("--beta_user", dest="beta_user", type="float", help="parameter beta_user", default=0.5)
	parser.add_option("--beta_place", dest="beta_place", type="float", help="parameter beta_place", default=0.5)
	parser.add_option("--ku", dest="Ku", type="int", help="number of user-topics", default=10)
	parser.add_option("--kp", dest="Kp", type="int", help="number of place-topics", default=10)
	parser.add_option("-n", dest="n", type="int", help="print the first n-most representive topics", default=10)
	parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=400)
	parser.add_option("-v", dest="verbose", action="store_true", help="Print the user-topics, place-topics and their interrelation", default=False)
	parser.add_option("--seed", dest="seed", type="int", help="random seed")

	(options, args) = parser.parse_args()


	#**2. Error Handling
	if options.filename:
		data = numpy.genfromtxt(options.filename,delimiter=',',dtype=None)
	else:
		 parser.error("need filename(-f)")
		
	if options.seed != None:
		numpy.random.seed(options.seed)

	#**3. Run Inference Algorithm
	w_z,a_z,topics,idx2vals,vals2idx=ylda.joint_lda(data,options.Ku,options.Kp,options.n,options.alpha,options.beta_user,options.beta_place,it=options.iteration,verbose=options.verbose)

