
# Core library
<!-- TOC -->

- [Core library](#core-library)
  - [Data structures](#data-structures)
    - [Array](#array)
    - [Array Handler](#array-handler)
  - [Debug](#debug)
  - [IO](#io)
  - [Math](#math)
  - [Memory](#memory)
  - [Meta](#meta)
  - [Types](#types)

<!-- /TOC -->

### Data structures
#### Array
The Array data structure is a dynamically allocated array in which the elements have template type `T` and the allocation, copy and set operations are handled according the memory specification dictated by the template type `Allocator`, providing the following set of features:
- Capable of handling data copies between different types of allocators, making it suitable for copies between different kind of devices ex.:
```
  Array<T,cpuAllocator> arrayCPU(size,dev);
  ...
  Array<T,gpuAllocator> arrayGPU(arrayCPU);
```
- Memory model independent which enables portability and extensibility for ex.:
```
  Array<T,amdAllocator> arrayAMD(size,dev);
  Array<T,nvidiaAllocator> arrayNVIDIA(size,dev);
  Array<T,mpiAllocator> arrayMPI(size,dev);
```
- Interface somewhat similar to the one found in the c++ std.

#### Array Handler
Is a light weight version of the `Array` data structure with the sole purpose of being used in places where the `Array` data structure is too complicated, like device code.

### Debug
Provides a series of functions to introduce with ease debugging operations to the code, specifically it provides:
- The `error` macro which in case of being evaluated to false generates a certain type of notification. In particular there is a possibility to select to emit only certain types of errors, ex.:
```
  #define DEBUG_LEVEL api_debug|runtime_debug
  ...
  error(condition,"Error message",API_ERROR,stderr_error) //May get evaluated
  error(condition,"Error message",RUNTIME_ERROR,throw_error) //May get evaluated
  error(condition,"Error message",MEMORY_ERROR,assert_error) //Never gets evaluated
```
- The `cuda_error` checks for errors in the last CUDA call.
- The class `cpu_timer` and `gpu_timer` provides accurate timers with great ease of use:
```
  cpu_timer timer;
  timer.start();
  timer.stop();
  double elapsed_time;
  elapsed_time<<timer;
```

### IO
Provides a series of functions for IO operations, like printing the `Array`, `std::vector`, and other data structures.
### Math
Provides a series of interfaces for elementary math operations specialized for _CUDA_, for example:
```
  __add__<RN>(1.1,0.9); // Add the numbers rounding to the nearest
  __add__<RU>(1.1,0.9); // Add the numbers rounding to + infinity
  __div__<FAST_MATH>(2.1f,1.2f); // Divide the numbers using the fast CUDA intrinsic __fdividef
```
### Memory
Provides a series of interfaces to allocate, set, copy and free memory both in CUDA and CPU, specifically it provides `cpuAllocator`, `gpuAllocator`, `managedAllocator` and `pinnedAllocator` allocators, which are compatible with the `Array` data structure.
### Meta
Provides a series of functions and types designed to aide in the use of the meta programming capabilities of c++1x, for example:
- Compile time `for` unrollers:
 ```
__unroll_gpu__(int,i,0,2,(vec_T&,const T*),(vec_T& r,const T* x),(r,x),(r[i]=static_cast<V>(x[i]);))
// The compiler will evaluate to:
r[0]=static_cast<V>(x[0]);
r[1]=static_cast<V>(x[1]);
 ```
- Non integer template arguments:
 ```
template <int var=3> f() ... \\ Is legal in c++
template <double var=3> f() ... \\ Is illegal in c++
template <typename var=CTA(double,3.14)> f() ... \\ Is legal and the value can be accessed via var::value
 ```

### Types
Provides statically allocated vector types compatible with device code, with template type `T`, size `N` and alignment `A` :
```
  Vector<double,2,16> v({3.14,2.71});
  v[0]+=v[1]; // Now v[0] holds 3.14+2.71
```
Is important to understand that the alignment `A` plays a vital role on how the compiler will load the values from RAM.
