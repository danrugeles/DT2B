#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Dual Topics to Bicluster inference using Collapsed Gibbs Sampling
# This code is available under the MIT License.
# (c)2016 Daniel Rugeles 
# daniel007[at]e.ntu.edu.sg / Nanyang Technological University


if __name__ == "__main__":

	import numpy
	import optparse
	import csv
	import dt2b

	#**1. Parse Settings  
	parser = optparse.OptionParser()
	parser.add_option("-f", dest="filename", help="filename of the dataset")
	parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
	parser.add_option("--beta_row", dest="beta_row", type="float", help="parameter beta_row", default=0.5)
	parser.add_option("--beta_column", dest="beta_column", type="float", help="parameter beta_column", default=0.5)
	parser.add_option("--ku", dest="Ku", type="int", help="number of row-topics", default=10)
	parser.add_option("--kp", dest="Kp", type="int", help="number of column-topics", default=10)
	parser.add_option("-n", dest="n", type="int", help="print the first n-most representive topics", default=10)
	parser.add_option("-i", dest="iteration", type="int", help="Maximum number of iterations", default=500)
	parser.add_option("-v", dest="verbose", action="store_false", help="Print the row-topics, column-topics and their interrelation", default=True)
	parser.add_option("--seed", dest="seed", type="int", help="random seed")

	(options, args) = parser.parse_args()


	#**2. Error Handling
	if options.filename:
		data = numpy.genfromtxt(options.filename,delimiter=',',dtype=None)
	else:
		 parser.error("need filename(-f). Try main.py -h for help")
		
	if options.seed != None:
		numpy.random.seed(options.seed)

	#**3. Run Inference Algorithm
	w_z,a_z,joint,idx2vals,vals2idx=dt2b.dt2b(data,options.Ku,options.Kp,options.n,options.alpha,options.beta_row,options.beta_column,it=options.iteration,verbose=options.verbose)


