#include "hpctimer.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

enum { N = 15000 };

/*последовательная версия*/
void matrix_vector_product(double *a, double *b, double *c, int m, int n) {
  for (int i = 0; i < m; i++) {
    c[i] = 0.0;
    for (int j = 0; j < n; j++)
      c[i] += a[i * n + j] * b[j];
  }
}

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n,
                               int stream) {
#pragma omp parallel num_threads(stream)
  {
    int nthreads = omp_get_num_threads();
    int threadid = omp_get_thread_num();
    int items_per_thread = m / nthreads;
    int lb = threadid * items_per_thread; // ниж. граница
    int ub = (threadid == nthreads - 1)
                 ? (m - 1)
                 : (lb + items_per_thread - 1); // верх. граница

    for (int i = lb; i < ub; i++) {
      c[i] = 0.0;
      for (int j = 0; j < n; j++) {
        c[i] = c[i] + a[i * n + j] * b[j];
      }
    }
  }
}

/*последовательная версия*/
double run_serial(int m, int n) {
  double *a, *b, *c;

  a = malloc(sizeof(*a) * m * n);
  b = malloc(sizeof(*b) * n);
  c = malloc(sizeof(*c) * m);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      a[i * n + j] = i + j;
    }
    c[i] = 0.0;
  }

  for (int j = 0; j < n; j++) {
    b[j] = j;
  }

  double t = hpctimer_getwtime();
  matrix_vector_product(a, b, c, m, n);
  t = hpctimer_getwtime() - t;

  free(a);
  free(b);
  free(c);

  return t;
}

double run_parallel(int m, int n, int stream) {
  double *a, *b, *c;

  a = malloc(sizeof(*a) * m * n);
  b = malloc(sizeof(*b) * n);
  c = malloc(sizeof(*c) * m);

#pragma omp parallel num_threads(stream)
  {
    int nthreads = omp_get_num_threads();
    int threadid = omp_get_thread_num();
    int items_per_thread = m / nthreads;
    int lb = threadid * items_per_thread;
    int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);

    for (int i = lb; i <= ub; i++) {
      for (int j = 0; j < n; j++)
        a[i * n + j] = i + j;
      c[i] = 0.0;
    }
  }

  for (int j = 0; j < n; j++) {
    b[j] = j;
  }

  double t = hpctimer_getwtime();
  matrix_vector_product_omp(a, b, c, m, n, stream);
  t = hpctimer_getwtime() - t;

  free(a);
  free(b);
  free(c);

  return t;
}

int main(int argc, char **argv) {
  int border = 25000;
  int step = 5000;

  for (int j = N; j <= border; j += step) {
    printf("%d\n", j);
    double t1 = run_serial(j, j);
    printf("T1 = %.6f\n", t1);

    for (int i = 2; i <= 8; i += 2) {

      double t2 = run_parallel(j, j, i);
      printf("T%d = %.6f\t", i, t2);

      printf("S = %.6lf\n", t1 / t2);
    }
    printf("\n");
  }

  return 0;
}
