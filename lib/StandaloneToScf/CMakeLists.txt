add_mlir_library(StandaloneStandaloneToScf
    StandaloneToScf.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRArithmetic
  MLIRControlFlow
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  )

