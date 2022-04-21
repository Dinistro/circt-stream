add_mlir_library(StandaloneStreamToHandshake
    StreamToHandshake.cpp

  DEPENDS
  StandaloneConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRControlFlow
  MLIRFunc

  CIRCTHandshake
  )

