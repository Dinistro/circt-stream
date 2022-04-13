add_mlir_dialect_library(MLIRStream
        StreamDialect.cpp
        StreamOps.cpp

        DEPENDS
        MLIRStreamOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
