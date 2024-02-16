#define _GNU_SOURCE
#include <stdlib.h>
#include <fftw3.h>
#include <time.h>
#include <stdint.h>	// uint64
#include <time.h>	// clock_gettime
#include <math.h>

#define BILLION 1000000000L

// Function to generate a random, complex signal
void gen_random_signal(double* reals, double* imags, int num_amps) {
    // Check for invalid input
    if (num_amps <= 0 || reals == NULL || imags == NULL) {
        fprintf(stderr, "Invalid input\n");
        exit(EXIT_FAILURE);
    }

    // Seed the random number generator
    srand((unsigned int)time(NULL));

    // Generate random values for probabilities
    double* probs = (double*)malloc(num_amps * sizeof(double));
    double total = 0.0;

    for (int i = 0; i < num_amps; ++i) {
        probs[i] = (double)rand() / RAND_MAX;
        total += probs[i];
    }

    // Normalize probabilities
    double total_recip = 1.0 / total;

    for (int i = 0; i < num_amps; ++i) {
        probs[i] *= total_recip;
    }

    // Generate random angles
    double* angles = (double*)malloc(num_amps * sizeof(double));

    for (int i = 0; i < num_amps; ++i) {
        angles[i] = 2.0 * M_PI * ((double)rand() / RAND_MAX);
    }

    // Generate complex values and fill the buffers
    for (int i = 0; i < num_amps; ++i) {
        double p_sqrt = sqrt(probs[i]);
        double sin_a, cos_a;

        double theta = angles[i];
        sin_a = sin(theta);
        cos_a = cos(theta);

        double re = p_sqrt * cos_a;
        double im = p_sqrt * sin_a;

        reals[i] = re;
        imags[i] = im;
    }

    // Free allocated memory
    free(probs);
    free(angles);
}


int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return EXIT_FAILURE;
    }

    long n = strtol(argv[1], NULL, 0);

    int N = 1 << n;

    // We don't count input mem allocation for RustFFT or PhastFT, so we omit
    // it from the timer here.
    fftw_complex* in = fftw_alloc_complex(N);

    uint64_t diff1;
	struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    fftw_plan p = fftw_plan_dft_1d(N, in, in, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
    clock_gettime(CLOCK_MONOTONIC, &end);
	diff1 = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;

    // Generate random complex signal using the provided function
    double* reals = (double*)malloc(N * sizeof(double));
    double* imags = (double*)malloc(N * sizeof(double));
    gen_random_signal(reals, imags, N);

    // Fill the FFT input array
    for (int i = 0; i < N; i++) {
        in[i][0] = reals[i];
        in[i][1] = imags[i];
    }
    free(reals);
    free(imags);

    uint64_t diff2;
	struct timespec start1, end1;
    clock_gettime(CLOCK_MONOTONIC, &start1);
    fftw_execute(p);
    clock_gettime(CLOCK_MONOTONIC, &end1);	/* mark the end1 time */
	diff2 = BILLION * (end1.tv_sec - start1.tv_sec) + end1.tv_nsec - start1.tv_nsec;

    uint64_t diff = (diff1 / 1000) + (diff2 / 1000);
	printf("%llu\n", (long long unsigned int) diff);

    fftw_free(in);
    fftw_destroy_plan(p);
    fftw_cleanup();

    return EXIT_SUCCESS;
}

