//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Conversion/Passes.h"
#include "Standalone/Dialect/Standalone/StandaloneDialect.h"
#include "Standalone/Dialect/Stream/StreamDialect.h"
#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;

  // TODO only register required dialects
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  circt::registerAllDialects(registry);
  circt::registerAllPasses();

  registry.insert<mlir::standalone::StandaloneDialect>();
  registry.insert<mlir::stream::StreamDialect>();

  mlir::standalone::registerConversionPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Standalone optimizer driver\n", registry));
}
