set(LIBS
        MLIRIR
        MLIRLLVMIR
        MLIRMemRef
        MLIROptLib
        MLIRParser
        MLIRFunc
        MLIRSupport
        MLIRTransforms
        MLIRSCF
        MLIRSCFToControlFlow

        MLIRStandalone
        MLIRStream

        CIRCTAffineToStaticLogic
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
        CIRCTStandardToStaticLogic
        CIRCTStaticLogicOps
        CIRCTSV
        CIRCTSVTransforms
        CIRCTTransforms

        StandaloneStandaloneToScf
        StandaloneStreamToHandshake
        )
add_llvm_executable(stream-opt stream-opt.cpp)

llvm_update_compile_flags(stream-opt)
target_link_libraries(stream-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(stream-opt)
