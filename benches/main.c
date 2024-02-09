#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <time.h>
#include <stdint.h>	// uint64
#include <time.h>	// clock_gettime

#define BILLION 1000000000L


int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return EXIT_FAILURE;
    }

    long n = strtol(argv[1], NULL, 0);

    int N = 1 << n;

    uint64_t diff1;
	struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    fftw_complex* in = fftw_alloc_complex(N);
    fftw_plan p = fftw_plan_dft_1d(N, in, in, FFTW_FORWARD, FFTW_ESTIMATE);
    clock_gettime(CLOCK_MONOTONIC, &end);
	diff1 = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;

    double a = 1.0;
    for (int i = 0; i < N; i++) {
        in[i][0] = ((double)rand()/(double)(RAND_MAX)) * a;
        in[i][1] = ((double)rand()/(double)(RAND_MAX)) * a;
    }

    uint64_t diff2;
	struct timespec start1, end1;
    clock_gettime(CLOCK_MONOTONIC, &start1);
    fftw_execute(p);
    clock_gettime(CLOCK_MONOTONIC, &end1);	/* mark the end1 time */
	diff2 = BILLION * (end1.tv_sec - start1.tv_sec) + end1.tv_nsec - start1.tv_nsec;

    uint64_t diff = diff1 + diff2;
	printf("%llu\n", (long long unsigned int) diff / 1000);

    fftw_free(in);
    fftw_destroy_plan(p);
    fftw_cleanup();

    return EXIT_SUCCESS;
}

