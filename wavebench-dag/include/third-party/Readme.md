# Third Party

<!-- TOC -->

- [Third Party](#third-party)
  - [SparseIO](#sparseio)
  - [Reorderings](#reorderings)

<!-- /TOC -->

## SparseIO
Provides functions to read and write sparse matrices in the _Rutherford-Boeing_ format, enabling the input of a bast collection sparse matrices from sites like [The SuiteSparse Matrix Collection ](https://sparse.tamu.edu/) and [Matrix Market](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/).
- `read(std::string filename)` returns the sparse matrix contained in the file `filename` in the container `MatrixCXS<T,IT,cpuAllocator,CCS>`.

## Reorderings
Provides functions to compute matrix reorderings to reduce _fill-in_ or increase the level of parallelism using algorithms like minimum degree reordering.
