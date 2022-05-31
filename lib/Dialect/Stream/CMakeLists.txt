add_mlir_dialect_library(CIRCTStreamStream
        StreamDialect.cpp
        StreamOps.cpp

        DEPENDS
        MLIRStreamOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
