import numpy as np 
from numpy.ctypeslib import ndpointer 
import ctypes 
from aux import join2

_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C') 

_dll = ctypes.CDLL('C/ylda_inference.so') 

_ylda_inference = _dll.ylda_inference
_ylda_inference.argtypes = [_doublepp,_doublepp,_doublepp,_doublepp,ctypes.c_int, ctypes.c_int,ctypes.c_int, ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_double] 
_ylda_inference.restype = None 


def createPointer(x):
	return (x.__array_interface__['data'][0] + np.arange(x.shape[0])*x.strides[0]).astype(np.uintp) 

#** Data must be dictionated
#** Column 1: User, Column 2: Place, Column 3: User-topic, Column 4: Place-topic
def inference(data,ite,alpha,beta_u,beta_p): 

	if type(data)!=type([]):
		data=data.tolist()
		
	data=np.array(data,dtype=np.intc)

	numusers=len(join2(data[:][:,[0]]))
	numplaces=len(join2(data[:][:,[1]]))
	numutopics=len(join2(data[:][:,[2]]))
	numptopics=len(join2(data[:][:,[3]]))
	uKu=join2(data[:][:,[0,2]]).astype(np.intc)
	pKp=join2(data[:][:,[1,3]]).astype(np.intc)
	KuKp=join2(data[:][:,[2,3]]).astype(np.intc)
	
	print "Number of Users",numusers
	print "Number of Places",numplaces,"\n"
	
	#** C-Domain -------------------------------------
	datapp = createPointer(data)
	uKupp = createPointer(uKu)
	pKppp = createPointer(pKp)
	KuKppp = createPointer(KuKp)

	it = ctypes.c_int(ite)     
	Ku = ctypes.c_int(numutopics)
	Kp = ctypes.c_int(numptopics)
	nusers = ctypes.c_int(numusers) 
	nplaces = ctypes.c_int(numplaces) 
	nrecs = ctypes.c_int(len(data)) 
	a = ctypes.c_double(alpha)
	bu = ctypes.c_double(beta_u)
	bp = ctypes.c_double(beta_p)
	
	
	_ylda_inference(datapp,uKupp,pKppp,KuKppp,it,Ku,Kp,nusers,nplaces,nrecs,a,bu,bp) 
	#** ------------------------------------------------
	
	return data



