add_mlir_library(StandaloneStandaloneToScf
    StandaloneToScf.cpp

  DEPENDS
  StandaloneConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRArithmetic
  MLIRControlFlow
  MLIRSupport
  MLIRTransforms
  )

