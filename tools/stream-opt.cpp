//===- stream-opt.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-stream/Conversion/Passes.h"
#include "circt-stream/Dialect/Stream/StreamDialect.h"
#include "circt-stream/Transform/Passes.h"
#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // TODO only register required dialects
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  mlir::registerSCFToControlFlowPass();

  // clang-format off
  registry.insert<
    circt::chirrtl::CHIRRTLDialect,
    circt::comb::CombDialect,
    circt::firrtl::FIRRTLDialect,
    circt::handshake::HandshakeDialect,
    circt::llhd::LLHDDialect,
    circt::hw::HWDialect,
    circt::seq::SeqDialect,
    circt::pipeline::PipelineDialect,
    circt::sv::SVDialect
  >();
  // clang-format on

  circt::registerAffineToPipelinePass();
  circt::registerConvertHWToLLHDPass();
  circt::registerConvertLLHDToLLVMPass();
  circt::registerExportSplitVerilogPass();
  circt::registerExportVerilogPass();
  circt::registerHandshakeRemoveBlockPass();
  circt::registerHandshakeToFIRRTLPass();
  circt::registerHandshakeToHWPass();
  circt::registerLowerFIRRTLToHWPass();
  circt::registerStandardToHandshakePass();

  circt::registerFlattenMemRefPass();
  circt::registerFlattenMemRefCallsPass();

  circt::firrtl::registerPasses();
  circt::llhd::initLLHDTransformationPasses();
  circt::seq::registerPasses();
  circt::sv::registerPasses();
  circt::handshake::registerPasses();
  circt::hw::registerPasses();

  registry.insert<circt_stream::stream::StreamDialect>();

  circt_stream::registerConversionPasses();
  circt_stream::registerTransformPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Stream optimizer driver\n", registry));
}
