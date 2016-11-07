all: dt2b_inference.c
	gcc -Wall -fPIC -c dt2b_inference.c 
	gcc -shared -Wl,-soname,dt2b_inference.so.1 -o dt2b_inference.so dt2b_inference.o `gsl-config --cflags --libs` 

clean:
	$(RM) dt2b_inference.so
	$(RM) dt2b_inference.o
