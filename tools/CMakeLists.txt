set(LIBS
        MLIRIR
        MLIRLLVMDialect
        MLIRMemRefDialect
        MLIROptLib
        MLIRParser
        MLIRFuncDialect
        MLIRSupport
        MLIRTransforms
        MLIRSCFDialect
        MLIRSCFToControlFlow

        CIRCTAffineToPipeline
        CIRCTAnalysisTestPasses
        CIRCTExportVerilog
        CIRCTFIRRTL
        CIRCTFIRRTLToHW
        CIRCTFIRRTLTransforms
        CIRCTHandshake
        CIRCTHandshakeToFIRRTL
        CIRCTHandshakeToHW
        CIRCTHandshakeTransforms
        CIRCTLLHD
        CIRCTLLHDToLLVM
        CIRCTLLHDTransforms
        CIRCTHW
        CIRCTHWToLLHD
        CIRCTHWTransforms
        CIRCTScheduling
        CIRCTSchedulingTestPasses
        CIRCTSeq
        CIRCTSeqTransforms
        CIRCTStandardToHandshake
        CIRCTPipelineOps
        CIRCTSV
        CIRCTSVTransforms
        CIRCTTransforms

        CIRCTStreamStream
        CIRCTStreamStreamToHandshake
        CIRCTStreamTransform
        )
add_llvm_executable(stream-opt stream-opt.cpp)

llvm_update_compile_flags(stream-opt)
target_link_libraries(stream-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(stream-opt)
