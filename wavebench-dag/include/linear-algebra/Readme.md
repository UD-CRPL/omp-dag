# Linear algebra
<!-- TOC -->

- [Linear algebra](#linear-algebra)
  - [Cholesky](#cholesky)
  - [Data Structures](#data-structures)
    - [LRTree](#lrtree)
  - [Matrix Formats](#matrix-formats)
    - [Matrix](#matrix)
    - [MatrixCXS](#matrixcxs)
    - [MatrixMap](#matrixmap)
  - [Matrix Operations](#matrix-operations)

<!-- /TOC -->

## Cholesky
Contains several versions for the Cholesky decomposition algorithm both dense and sparse. Specifically it contains:
- `cholesky(MatrixHandler<T,size_t>& R)` which computes the Cholesky decomposition for a dense matrix.
- `choleskyLeftLooking(MatrixCXSHandler<T,IT> L,MatrixCXSHandler<T,IT> A,...)` which computes the sparse Cholesky decomposition.

## Data Structures
Contains a series of data structures needed to compute the sparse Cholesky decomposition, in particular it contains:

#### LRTree
The class `LRTree` is a left child right sibling forest, supporting the basic tree instructions `insert`, `erase` and `find`. In particular it is compatible with device memory. It's usage is:
```
  LRTree<KeyType,ValueType,Allocator> tree;
  auto tmp=tree.insert(1); // Inserts a node with key 1 into the root of the forest.
  tree.insert(2,tmp);      // Inserts a node with key 2 with parent 1.
  tree.insert(3,tmp);      // Inserts a node with key 3 with parent 1.
  tree.insert(4);         //  Inserts a node with key 4 into the root of the forest.
  Now the forest holds:
        1              4
    2       3
```

## Matrix Formats
  Contains a set of data structures to hold sparse and dense matrices, and routines to `convert` between those formats. Specifically:

#### Matrix
Matrix is a dinamically allocated container for dense matrices, with the capability of performing copies between the `HOST` and `DEVICE`, and the option of specifying a pitch for all rows -enabling better cache performance.
```
  Matrix<double,cpuAllocator,128> Acpu(100,100); // Allocates a 100x100 matrix of type double on the HOST in which each row is aligned to 128 bytes.
  Acpu(0,0)=1; // Now Acpu holds a 1 in the entry 1,1
  Matrix<double,gpuAllocator,128> Agpu(Acpu); // Creates a copy of Acpu on the DEVICE.
```
As is the case with  `ArrayHandler` [^coreReadme], there is  a `MatrixHandler<double>` specifically deigned to use the data structure in device code.

[^coreReadme]: See [../core/Readme.md](../core/Readme.md)

#### MatrixCXS
Matrix is a dinamically allocated container for sparse matrices in compressed format, with the capability of performing copies between the `HOST` and `DEVICE`. In particular it handles the formats coordinate list `COO`, compressed row storage `CRS` and compressed column storage `CCS`.

```
  MatrixCXS<double,int,cpuAllocator,CRS> Acpu(pow(10,10),100); // Stores a pow(10,10) x pow(10,10) matrix with 100 non zero values on the HOST.
  MatrixCXS<double,int,gpuAllocator,CRS> Agpu(Acpu); // Creates a copy of Acpu on the GPU.
```
As is the case with  `MatrixHandler`, there is  a `MatrixCXSHandler<double,int>` specifically deigned to use the data structure in device code.

#### MatrixMap
Is a convenient sparse data structure based on `std::map`, which eases the construction of sparse matrices in compressed format and allows naive versions of complicated sparse algorithms like sparse GEMM.
```
  MatrixMap<double> A(pow(10,10)); // Creates an empty matrix of dimensions pow(10,10) x pow(10,10).
  A(0,0)=1;
  A(100,50)=1;
  MatrixCXS<double,int,cpuAllocator,CRS> Acxs;
  convert(Acxs,A); //Now Acxs holds a sparse matrix of dimensions pow(10,10) x pow(10,10), with non zero values on the entries 1,1 and 101,51.
```

## Matrix Operations
Contains a series of basic matrix operations like `transpose` and `multiply`, necessary to verify the correctness of the Cholesky algorithm. In particular it contains:
- `eliminationTree` which creates the elimination tree for a sparse matrix in compressed format.
- `colPattern`  which computes the non zero pattern of the Cholesky decomposition.
