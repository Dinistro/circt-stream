add_mlir_library(CIRCTStreamTransform
    CustomBufferInsertion.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils

  DEPENDS
  CIRCTStreamTransformPassIncGen
)

