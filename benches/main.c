#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <time.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return EXIT_FAILURE;
    }
    long n = strtol(argv[1], NULL, 0);
    printf("%ld\n", n);

    int N = 1 << n;
    fftw_complex* in = fftw_alloc_complex(N);

    double a = 1.0;
    for (int i = 0; i < N; i++) {
        in[i][0] = ((double)rand()/(double)(RAND_MAX)) * a;
        in[i][1] = ((double)rand()/(double)(RAND_MAX)) * a;
    }

    double tic = clock();
    fftw_plan p = fftw_plan_dft_1d(N, in, in, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    double toc = clock();
    double elapsed = ((double)(toc - tic) / CLOCKS_PER_SEC) * 1000000;

    printf("%f\n", elapsed);
    fftw_free(in);
    fftw_destroy_plan(p);
    fftw_cleanup();

    return EXIT_SUCCESS;
}

