all: ylda_inference.c
	gcc -Wall -fPIC -c ylda_inference.c 
	gcc -shared -Wl,-soname,ylda_inference.so.1 -o ylda_inference.so ylda_inference.o `gsl-config --cflags --libs` 

clean:
	$(RM) ylda_inference.so
	$(RM) ylda_inference.o
