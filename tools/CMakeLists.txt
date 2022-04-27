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
        CIRCTStandardToHandshake
        CIRCTStandardToStaticLogic
        CIRCTStaticLogicOps
        CIRCTSV
        CIRCTSVTransforms
        CIRCTTransforms

        StandaloneStandaloneToScf
        StandaloneStreamToHandshake
        )
add_llvm_executable(standalone-opt standalone-opt.cpp)

llvm_update_compile_flags(standalone-opt)
target_link_libraries(standalone-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(standalone-opt)
