#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Dual Topics to Bicluster using Collapsed Gibbs Sampling
# This code is available under the MIT License.
# (c)2016 Daniel Rugeles 
# daniel007[at]e.ntu.edu.sg / Nanyang Technological University


import numpy as np
from aux import dictionate,join2,addrandomtopic
import math
import dt2b_c
np.set_printoptions(precision=0)
import time

__author__ = "Daniel Rugeles"
__copyright__ = "Copyright 2016, DT2B"
__credits__ = ["Daniel Rugeles"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Daniel Rugeles"
__email__ = "daniel007@e.ntu.edu.sg"
__status__ = "Released"
	

#--DT2B ---------------------------------------------------------
"""Runs DT2B over the given data
data = 2D Numpy array  with structure <Row,column,zu,zp>
Ku = Number of Row-topics
Kp = Number of Column-topics
it = Number of iterations
"""
#-------------------------------------------------------------------
def dt2b(dt2bdata,Ku,Kp,n,alpha,beta_row,beta_column,it,verbose=True):

	def init(data,Ku,Kp):
	
		data=np.array(data)
		assert(data.shape[1]==2)
	
		datadt2b=np.zeros((len(data),4),dtype='|S20')
		for idx,row in enumerate(datadt2b):
			row[0]=data[idx][0]
			row[1]=data[idx][1]
	
		datadt2b=addrandomtopic(datadt2b,Ku,-2)
		datadt2b=addrandomtopic(datadt2b,Kp,-1)
	
		return datadt2b

	
	#--printTopics------------------------------------------------------
	"""Print Topics top 't' topics given conditional distribution given the topic
	p_z = ditribution given topic 
	t = Top T topics
	"""
	#------------------------------------------------------------
	def printColumnTopics(p_z,t):
		for idx,z in enumerate(p_z):
			print "Topic",idx,'- evidence'
			for topic,evidence in zip(np.argsort(z)[::-1][:t],np.sort(z)[::-1][:t]):
				print idx2vals[1][topic],int(evidence)
			print ""
			
	def printRowTopics(p_z,t):
		for idx,z in enumerate(p_z):
			print "Topic",idx,'- evidence'
			for topic,evidence in zip(np.argsort(z)[::-1][:t],np.sort(z)[::-1][:t]):
				print idx2vals[0][topic],int(evidence)
			print ""

	def	printTopics(mdata,verbose): 		
		
		print "Row Topics\n"
		printRowTopics(join2(processed_data[:][:,[2,0]]),n)
		print "------------------------------"
			
		print "Column Topics\n"
		printColumnTopics(join2(processed_data[:][:,[3,1]]),n)
		print "------------------------------"
	
		print "Topic Interrelation\n Evidence of the relationship between the row-topics and column-topics\n" 
		print join2(processed_data[:][:,[2,3]]).astype(np.int)
	
		print "------------------------------"
		
		
		
	"""-----------------*
	*                   *
	* |\/|  /\  | |\ |  * 
	* |  | /  \ | | \|  *
	*                   *
	*----------------"""

	dt2bdata=np.asarray(dt2bdata)
	dt2bdata=init(dt2bdata,Ku,Kp)
	
	print "Processing the data ..."
	dt2bdata,idx2vals,vals2idx=dictionate(dt2bdata)	
	dt2bdata=dt2bdata.astype(np.int)

	print "Running the inference process ..."
		
	start=time.time()
	
	processed_data = dt2b_c.inference(dt2bdata,it)
	
	print 'Inference Took:',time.time()-start,'seconds'
	
	if verbose:
		printTopics(processed_data,verbose)
	
	columns_w_z=join2(processed_data[:][:,[3,1]])
	rows_w_z=join2(processed_data[:][:,[2,0]])
	joint=join2(processed_data[:][:,[2,3]])

	return columns_w_z,rows_w_z,joint,idx2vals,vals2idx

