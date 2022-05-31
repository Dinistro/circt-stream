add_mlir_library(CIRCTStreamStreamToHandshake
    StreamToHandshake.cpp

  DEPENDS
  CIRCTStreamConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRControlFlow
  MLIRFunc

  CIRCTHandshake
  )

