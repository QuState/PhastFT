CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native
LIBS = -lfftw3 -lm

program: main.c
	$(CC) $(CFLAGS) -o bench_fftw main.c $(LIBS)

clean:
	rm -f program

