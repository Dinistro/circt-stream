get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRStandalone

        # TODO drop unnessary dependencies
        CIRCTAffineToStaticLogic
        CIRCTAnalysisTestPasses
        CIRCTCalyx
        CIRCTCalyxToHW
        CIRCTCalyxTransforms
        CIRCTESI
        CIRCTExportVerilog
        CIRCTFIRRTL
        CIRCTFIRRTLToHW
        CIRCTFIRRTLTransforms
        CIRCTFSM
        CIRCTHandshake
        CIRCTHandshakeToFIRRTL
        CIRCTHandshakeToHW
        CIRCTHandshakeTransforms
        CIRCTLLHD
        CIRCTLLHDToLLVM
        CIRCTLLHDTransforms
        CIRCTMoore
        CIRCTMooreToCore
        CIRCTMSFT
        CIRCTHW
        CIRCTHWToLLHD
        CIRCTSCFToCalyx
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
        )
add_llvm_executable(standalone-opt standalone-opt.cpp)

llvm_update_compile_flags(standalone-opt)
target_link_libraries(standalone-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(standalone-opt)
