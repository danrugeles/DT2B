import numpy as np 
from numpy.ctypeslib import ndpointer 
import ctypes 
from aux import join2

_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C') 
_dll = ctypes.CDLL('C/DT2B/dt2b_inference.so') 

_dt2b_inference = _dll.dt2b_inference
_dt2b_inference.argtypes = [_doublepp,_doublepp,_doublepp,_doublepp,ctypes.c_int, ctypes.c_int,ctypes.c_int, ctypes.c_int,ctypes.c_int,ctypes.c_int] 
_dt2b_inference.restype = None 


def createPointer(x):
	return (x.__array_interface__['data'][0] + np.arange(x.shape[0])*x.strides[0]).astype(np.uintp) 

#** Data must be dictionated
#** Column 1: one row, Column 2: one column, Column 3: Row-topic, Column 4: Column-topic
def inference(data,ite): 

	if type(data)!=type([]):
		data=data.tolist()
		
	data=np.array(data,dtype=np.intc)

	numrows=len(join2(data[:][:,[0]]))
	numcolumns=len(join2(data[:][:,[1]]))
	numrtopics=len(join2(data[:][:,[2]]))
	numctopics=len(join2(data[:][:,[3]]))
	uKu=join2(data[:][:,[0,2]]).astype(np.intc)
	pKp=join2(data[:][:,[1,3]]).astype(np.intc)
	KuKp=join2(data[:][:,[2,3]]).astype(np.intc)
	
	print "Number of Rows",numrows
	print "Number of Columns",numcolumns,"\n"
	
	#** C-Domain -------------------------------------
	datapp = createPointer(data)
	uKupp = createPointer(uKu)
	pKppp = createPointer(pKp)
	KuKppp = createPointer(KuKp)

	it = ctypes.c_int(ite)     
	Ku = ctypes.c_int(numrtopics)
	Kp = ctypes.c_int(numctopics)
	nrows = ctypes.c_int(numrows) 
	ncolumns = ctypes.c_int(numcolumns) 
	nrecs = ctypes.c_int(len(data)) 
	
	_dt2b_inference(datapp,uKupp,pKppp,KuKppp,it,Ku,Kp,nrows,ncolumns,nrecs) 
	#** ------------------------------------------------
	
	return data



