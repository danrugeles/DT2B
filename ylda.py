#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Yin Yang Latent Dirichlet Allocation using Collapsed Gibbs Sampling
# This code is available under the MIT License.
# (c)2015 Daniel Rugeles 
# daniel007[at]e.ntu.edu.sg / Nanyang Technological University


import numpy as np
from aux import dictionate,join2,addrandomtopic
import math
import joint_c
np.set_printoptions(precision=0)

__author__ = "Daniel Rugeles"
__copyright__ = "Copyright 2015, YLDA"
__credits__ = ["Daniel Rugeles"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Daniel Rugeles"
__email__ = "daniel007@e.ntu.edu.sg"
__status__ = "Released"


#--Joint LDA---------------------------------------------------------
"""Runs Joint LDA over the given data
data = 2D Numpy array  with structure <User,place,zu,zp>
Ku = Number of User-topics
Kp = Number of Place-topics
it = Number of iterations
"""
#-------------------------------------------------------------------
def joint_lda(ldadata,Ku,Kp,n,alpha,beta_user,beta_place,it,verbose=True):

	def init(data,Ku,Kp):
	
		data=np.array(data)
		assert(data.shape[1]==2)
	
		datalda=np.zeros((len(data),4),dtype='|S20')
		for idx,row in enumerate(datalda):
			row[0]=data[idx][0]
			row[1]=data[idx][1]
	
		datalda=addrandomtopic(datalda,Ku,-2)
		datalda=addrandomtopic(datalda,Kp,-1)
	
		return datalda

	
	#--printTopics------------------------------------------------------
	"""Print Topics top 't' topics given conditional distribution given the topic
	p_z = ditribution given topic 
	t = Top T topics
	"""
	#------------------------------------------------------------
	def printPlaceTopics(p_z,t):
		for idx,z in enumerate(p_z):
			print "Topic",idx
			for topic,evidence in zip(np.argsort(z)[::-1][:t],np.sort(z)[::-1][:t]):
				print idx2vals[1][topic],int(evidence)
			print ""
			
	def printUserTopics(p_z,t):
		for idx,z in enumerate(p_z):
			print "Topic",idx
			for topic,evidence in zip(np.argsort(z)[::-1][:t],np.sort(z)[::-1][:t]):
				print idx2vals[0][topic],int(evidence)
			print ""

	def	printTopics(mdata,verbose): 		
		
		print "User Topics\n"
		printUserTopics(join2(processed_data[:][:,[2,0]]),n)
		print "---------------"
			
		print "Place Topics\n"
		printPlaceTopics(join2(processed_data[:][:,[3,1]]),n)
		print "---------------"
	
		print "Topic Interrelation\n  Rows: User-topics\n  Column: Place-topics\n" 
		print join2(processed_data[:][:,[2,3]]).astype(np.int)
		print "---------------"
		
		
	"""-----------------*
	*                   *
	* |\/|  /\  | |\ |  * 
	* |  | /  \ | | \|  *
	*                   *
	*----------------"""

	ldadata=np.asarray(ldadata)
	ldadata=init(ldadata,Ku,Kp)
	
	print "Processing the data ..."
	ldadata,idx2vals,vals2idx=dictionate(ldadata)	
	ldadata=ldadata.astype(np.int)

	print "Running the inference process ..."
	processed_data = joint_c.inference(ldadata,it,alpha,beta_user,beta_place)
	
	if verbose:
		printTopics(processed_data,verbose)
	
	places_w_z=join2(processed_data[:][:,[3,1]])
	users_w_z=join2(processed_data[:][:,[2,0]])
	joint=join2(processed_data[:][:,[2,3]])

	return places_w_z,users_w_z,joint,idx2vals,vals2idx


	
	


