.PHONY : all
all : task2_low task2_high

task2_low : task2_low.c
	gcc -o task2_low task2_low.c -O2 -lm -lgsl -lgslcblas

task2_high : task2_high.c
	gcc -o task2_high task2_high.c -O2 -lm -lgsl -lgslcblas

clean : 
	touch *.c
