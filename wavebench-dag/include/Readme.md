# OMP-DAG
<!-- TOC -->

- [OMP-DAG](#ompdag)
  - [Core library](#core-library)
  - [Linear algebra](#linear-algebra)
  - [Third party](#third-party)
  - [Tests](#tests)

<!-- /TOC -->
## Core library
Is a project independent library aimed at easing the build of new libraries .It is included with:

```
  #include "core/core.h"
  using namespace __core__;
```
For more information see [core/Readme.md](core/Readme.md).

## Linear algebra

Is the main set of functions aimed at computing the Cholesky decomposition. It is included with:

```
  #include "linear-algebra/linear-algebra.h"
  using namespace __core__;
```
For more information see [linear-algebra/Readme.md](linear-algebra/Readme.md).

## Third party

Is a set of wrappers to popular libraries like _Suite Sparse_ and _METIS_ needed in certain algorithms or test cases. It is included with:

```
  #include "third-party/third-party.h"
  using namespace __third_party__;
```
For more information see [third-party/Readme.md](third-party/Readme.md).

