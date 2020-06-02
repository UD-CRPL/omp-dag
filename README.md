# Wave-DAG

Wave-DAG is a preprocessor for accelerating structured wavefront patterns on multi core architectures built on top of OpenMP 5.0, ensuing future portability across platforms.

The test cases used in Wave-DAG were taken from the Wavebench project. For more information about Wavebench see [wavebench/README.md](wavebench/README.md).

For more information about the underlying library see [wavebench-dag/include/Readme.md](wavebench-dag/include/Readme.md).


## Example

Basic Smith-Waterman serial code.
```
  for(int i = 1; i < n; ++i)
    for(int j = ; j < m; ++j) {
        int score = (A[i - 1] == B[j - 1])? match : miss;
        M[i * m + j] = max(M[(i - 1) * m + (j - 1)] + score,
                       max(M[i * m +(j - 1)] + gap, M[(i - 1) * m + j] + gap));
    }
```

Basic Smith-Waterman with Wave-DAG pragmas added.
```
#pragma omp dag coarsening(block, 512, 512)
  for(int i = 1; i < n; ++i)
    for(int j = ; j < m; ++j) {
#pragma omp dag task depend({(i + 1) * m + j + 1,((i + 1) < n) && ((j + 1) < m)}, \
                            {(i + 1) * m + j,(i + 1) < n},                \
                            {i * m + j + 1,(j + 1) < m})
      {
        int score = (A[i - 1] == B[j - 1]) ? match : miss;
        M[i * m + j] = max(M[(i - 1) * m + (j - 1)] + score,
                       max(M[i * m +(j - 1)] + gap, M[(i - 1) * m + j] + gap));
      }
    }
```
The resulting code then will be preprocessod by the Wave-DAG preprocessor and output a C++ OpenMP 5.0 compliant parallel code.

## Building and runninng
To build Wave-DAG you need GCC 9.2 and in order to run the original wavebench code you need a working PGI compiler.
To compile the tests use:
```
make omp     # Compile the Wave-DAG version of the test cases
make serial  # Compile a serial version of the test cases
make acc-gpu # Compile the Wavebench GPU version of the test cases
```
To run the samples use:
```
bash run.sh
```

## License
For more information see [LICENSE](LICENSE).
