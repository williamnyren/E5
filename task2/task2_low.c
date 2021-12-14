#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_randist.h>
/* #define SiO2_MASS 60.0843
#define MASS_CONVERTION_FACTOR 9649.0 */
#define KB 1.380649*1e-2 //	[ag*(mm)^2/(ms)^2])// 8.617333262145*1e-5 // eV/K, 	
#define dims 1

/* Simulation parameters */
const double dt = 0.001; /* ms = milli sec = 10^-3 sec */
const double tau = 147.3*1e-3; /* ms = milli sec */ /* LOW: 147.3 \mu sec, HIGH: 48.5 \mu sec */ 
const double friction_coefficient = 1.0/tau;
const double omega_0 = 2.0 * M_PI * 3.1; /* 1/ms */
const double T = 297.0; /* K */
const long int sim_time = 100;
const long int sim_steps = (long int)( sim_time * 1000) / ((long int)(dt*1e3));
const long int n_timesteps = 1 * sim_steps; /* X * [production group steps] */

/* Properties of Brownian particle */
typedef struct Brownian_particle{
    double r;
    double v;
    double a;
    double m;
    double m_inv;
}Brownian_particle;



void velocity_verlet_algorithm(double *, double *, double, 
                            Brownian_particle, 
                            gsl_rng *, gsl_rng *, gsl_rng *);


void calc_Cl(double *);

int main(){

    const double c_0 = 1.0/exp(friction_coefficient * dt);

    /* Initiate particle */
    Brownian_particle p1;
    p1.r = 0.1 * 1e-3;
    p1.v = 2.0 * 1e-3;
    p1.a = 0;
    p1.m =  30134022.3516; /* ag = atto gram = 10^-18g */  /* SiO2_MASS/MASS_CONVERTION_FACTOR; */
    p1.m_inv = 1.0/p1.m;

    /* Random number generator */
    const gsl_rng_type *randType1 = gsl_rng_default;
    unsigned long int seed1 = n_timesteps*4;
    gsl_rng *rand_gen1 = gsl_rng_alloc(randType1);
    gsl_rng_set( rand_gen1, seed1);
    gsl_ran_ugaussian( rand_gen1 );

    const gsl_rng_type *randType2 = gsl_rng_default;
    unsigned long int seed2 = n_timesteps*2;
    gsl_rng *rand_gen2 = gsl_rng_alloc(randType2);
    gsl_rng_set( rand_gen2, seed2);
    gsl_ran_ugaussian( rand_gen2 );

    const gsl_rng_type *randType3 = gsl_rng_default;
    unsigned long int seed3 = 0;
    gsl_rng *rand_gen3 = gsl_rng_alloc(randType3);
    gsl_rng_set( rand_gen3, seed3);
    gsl_ran_ugaussian( rand_gen3 );

    /* Just for testing the random numbers */
    /* run the program with the command " ./task1 > ugaussian_results.dat "
        to save prinf(...) output to file */
    /*
    long int iter = 10000;
    for( long int ix = 0; ix < iter; ++ix ){
            printf("%lf\n", gsl_ran_ugaussian( rand_gen ));
    }
    */
    /* Malloc to store trajectory in phase space */
    double *v_phase = (double *)malloc( n_timesteps * sizeof(double) );

    double *r_phase = (double *)malloc( n_timesteps * sizeof(double) );

    /* Relaxation */
    /*velocity_verlet_algorithm( r_phase, v_phase, c_0, p1, 
    rand_gen1, rand_gen2, rand_gen3);
    velocity_verlet_algorithm( r_phase, v_phase, c_0, p1, 
    rand_gen1, rand_gen2, rand_gen3);*/

    int trajs = 5;
    char ** fnames = (char **)malloc( trajs * sizeof(char*) );
    fnames[0] = "0_low.dat";
    fnames[1] = "1_low.dat";
    fnames[2] = "2_low.dat";
    fnames[3] = "3_low.dat";
    fnames[4] = "4_low.dat";

    for(int traj = 0; traj < trajs; ++traj)
    {
        p1.r = 0.1 * 1e-3;
        p1.v = 2.0 * 1e-3;
        /* Run velocity verlet on a particle */
        velocity_verlet_algorithm( r_phase, v_phase, c_0, p1, 
        rand_gen1, rand_gen2, rand_gen3);

        /* write to file */
        /* CHANGE TO A cahr ** AND ITER. OVER IT */
        FILE *fp = fopen(fnames[traj], "w");
        fprintf(fp, "r_phase, v_phase");
        for( long int tx = 0; tx < n_timesteps; ++tx){
            fprintf(fp, "\n%.16f,%.16f", r_phase[tx], v_phase[tx]);
        }
        fclose(fp);
    }
    /* calc_Cl(v_phase); */

    free(r_phase);
    free(v_phase);
    gsl_rng_free (rand_gen1);
    gsl_rng_free (rand_gen2);
    gsl_rng_free (rand_gen3);

}



void velocity_verlet_algorithm( double * r_phase, double * v_phase, 
            double c_0, Brownian_particle p, 
            gsl_rng *rand_gen1, gsl_rng *rand_gen2, gsl_rng *xi)
{       
    double gauss_rand1, gauss_rand2, gauss_rand_xi;
    double v_th = sqrt(KB*T*p.m_inv);
    for(long int tx = 0; tx < n_timesteps; ++tx){
        
        /* half-step */
        gauss_rand1 = gsl_ran_ugaussian( rand_gen1 );

        p.v = 0.5*p.a * dt 
                    + sqrt(c_0) * p.v 
                    + v_th * sqrt(1 - c_0) * gauss_rand1;

        p.r = p.r + p.v * dt;

        /* Calculate acc */
        gauss_rand_xi = gsl_ran_ugaussian( xi );
        p.a = -(omega_0*omega_0) * p.r 
                    - friction_coefficient * p.v
                    + (2.0 * friction_coefficient * KB * T * p.m_inv)*gauss_rand_xi;

        /* Full-step */
        gauss_rand2 = gsl_ran_ugaussian( rand_gen2 );

        p.v = 0.5*sqrt(c_0) * p.a * dt 
                    + sqrt(c_0) * p.v 
                    + v_th * sqrt(1 - c_0) * gauss_rand2;

        /* Save trajectory */
        r_phase[tx] = p.r;
        
        v_phase[tx] = p.v;
    }
}


void calc_Cl(double * A)
{
    long int tau_dt_ratio = 3;
    long int M = (long int)(n_timesteps/tau_dt_ratio);
    long int l_stop = (long int)(M/100);
    double * Cl = (double *)malloc(l_stop * sizeof(double));
    double * t = (double *)malloc(l_stop * sizeof(double));
    double sum, Am, Aml, Cl_max;
    Cl_max = 0;
    for(long int lx = 0; lx < l_stop; ++lx){
        sum = 0;
        for(long int mx = 0; mx < M - lx; ++mx){
            Aml = A[(mx+lx)*tau_dt_ratio];
            Am = A[mx*tau_dt_ratio];
            sum += Aml*Am;
        }
        Cl[lx] = sum/(double)(M-lx);
        Cl_max = (Cl[lx] > Cl_max)*Cl[lx] + (Cl[lx] <= Cl_max)*Cl_max;
        t[lx] = dt*(double)(lx * tau_dt_ratio);
        /*if(lx % 1000 == 0)
            printf("%lf procent done.\n", ((double)(lx))/((double)l_stop));*/
    }

    for(long int lx = 0; lx < l_stop; ++lx)
    {
        Cl[lx] = Cl[lx]/Cl_max;        
    }
    /* write to file */
    char * fname = "Cl_low.dat";
    FILE *fp = fopen(fname, "w");
    fprintf(fp, "t(l), Cl");
    for( long int lx = 0; lx < l_stop; ++lx){
	    fprintf(fp, "\n%.16f,%.16f", t[lx], Cl[lx]);
    }
    fclose(fp);
    free(Cl);
    free(t);
}