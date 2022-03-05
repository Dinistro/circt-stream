add_mlir_library(StandaloneStandaloneToScf
    StandaloneToScf.cpp

  DEPENDS
  StandaloneConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRArithmetic
  MLIRControlFlow
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  )

