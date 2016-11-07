#include <stdlib.h> 
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <string.h>

//** Init Multinomial 
gsl_rng* init_gsl_rng(){
		gsl_rng* r;
 		const gsl_rng_type* T2;
    
    srand(time(NULL));
    unsigned int seed;
    seed= rand(); 

    gsl_rng_env_setup();

    T2 = gsl_rng_default;
    r = gsl_rng_alloc (T2);
    gsl_rng_set (r, seed);
	
		return r;    
}


unsigned int substract(unsigned int val){
	if(val>0)
		return val-1;
	else
		return 0;
}



//** Data must be dictionated
//** Doc-Word-topic
void dt2b_inference(int** data , unsigned int** uKu, unsigned  int** pKp, unsigned  int** KuKp, const int it, const int Ku,const int Kp, const int nusers, const int nplaces, const int nrecs){

	float l,prev_l,prob=0;
	
 	//** Allocate Dynamic Tables
	unsigned int zu[Ku];
	memset( zu, 0, Ku*sizeof(unsigned int) );
	unsigned int zp[Kp];
	memset( zp, 0, Kp*sizeof(unsigned int) );
	double p_zuzp[Ku][Kp];
	


  //** Initialization
	gsl_rng * r;
	r=init_gsl_rng();
	

	//** Initialize Dynamic Tables
	unsigned int oneutopic,oneptopic;
	size_t rowid;
	for(rowid=0;rowid<nrecs;rowid++){
			oneutopic=data[rowid][2];
			oneptopic=data[rowid][3];
			zu[(size_t)oneutopic]++;
			zp[(size_t)oneptopic]++;
	}
	
	//** Iterate
	size_t i,s,kp,ku;
	int curr_u,curr_p,curr_ku,curr_kp; 
	for(i=0;i<it;i++){		
		l=0;
		//printf("iteration %zu\n",i);
		for(rowid=0;rowid<nrecs;rowid++){

			//** Discount
			curr_u=data[rowid][0];
			curr_p=data[rowid][1];
			curr_ku=data[rowid][2];
			curr_kp=data[rowid][3];
			
			uKu[curr_u][curr_ku]=substract(uKu[curr_u][curr_ku]);
			pKp[curr_p][curr_kp]=substract(pKp[curr_p][curr_kp]);
			KuKp[curr_ku][curr_kp]=substract(KuKp[curr_ku][curr_kp]);
			zu[curr_ku]=substract(zu[curr_ku]);
			zp[curr_kp]=substract(zp[curr_kp]);
			
			
			//** Obtain Posterior
			prob=0;
			for(ku=0;ku<Ku;ku++){
				for(kp=0;kp<Kp;kp++){
					p_zuzp[ku][kp]=((KuKp[ku][kp]+0.5)*(uKu[curr_u][ku]+0.5)*(pKp[curr_p][kp]+0.5))/((zu[ku]+0.5*nusers)*(zp[kp]+0.5*nplaces));			
					prob+=p_zuzp[ku][kp];	
				}
			}
			l+=log(prob);
					
		 	
			//** Obtain one sample
    	unsigned int samples[Ku*Kp];
    	double* p_z = (double*) p_zuzp;
    	
   		gsl_ran_multinomial(r, Ku*Kp, 1, p_z, samples);
   			
 			size_t chosen_ku,chosen_kp;
 			for(s=0;s<Ku*Kp;s++){
 			
 				if (samples[s]>0){
 					
 					//get two indexes from one
 					chosen_ku=(unsigned int) (s/Kp);
 					chosen_kp=(unsigned int) (s%Kp);
 					
 					break;
 				}
 			}
   		
   		
 			//** Update
 			data[rowid][2]=chosen_ku;
 			data[rowid][3]=chosen_kp;
 			uKu[curr_u][chosen_ku]+=1;
 			pKp[curr_p][chosen_kp]+=1;
 			KuKp[chosen_ku][chosen_kp]+=1;
 			zu[chosen_ku]+=1;
 			zp[chosen_kp]+=1;
 			
		}
		
	if (i==0)
		prev_l=l;
	
	//** Check convergence
		if (l<prev_l && i>50){
			printf("Stopped at iteration %zu",i);
			break;
		}
				
		printf("Likelihood at iteration %zu: %f \n",i,l);
		
	}

	gsl_rng_free (r);   	
}
	
 
