#ifndef __ENUM_DEFINITIONS_MATRIX_FORMATS_H__
#define __ENUM_DEFINITIONS_MATRIX_FORMATS_H__

namespace __core__ {
namespace __linear_algebra__ {
namespace __matrix_formats__ {
enum __sparse_cxs_formats_enum__ {
	COO,
	CRS,
	CCS
};
typedef __sparse_cxs_formats_enum__ __SparseCXSFormat__;
typedef __SparseCXSFormat__ SparseCXSFormat;
}
}
}
#endif
