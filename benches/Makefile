CC = gcc
CFLAGS = -Wall -Wextra -Werror -O3
LIBS = -lfftw3 -lm

bench_fftw: main.c
	$(CC) $(CFLAGS) -o bench_fftw main.c $(LIBS)

clean:
	rm -f bench_fftw

