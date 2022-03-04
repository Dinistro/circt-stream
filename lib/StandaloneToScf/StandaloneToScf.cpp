//===- StandaloneToScf.cpp - Translate Standalone into SCF ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Standalone to Scf Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneToScf/StandaloneToScf.h"

#include "../PassDetail.h"
#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::standalone;

namespace {

struct NegToZeroLowering : public OpRewritePattern<NegToZeroOp> {
  using OpRewritePattern<NegToZeroOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NegToZeroOp negToZeroOp,
                                PatternRewriter &rewriter) const override {
    Location loc = negToZeroOp.getLoc();
    Type i32 = rewriter.getType<IntegerType>(32);

    Value const0 = rewriter.create<arith::ConstantIntOp>(loc, 0, i32);
    auto cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                               const0, negToZeroOp.input());
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        negToZeroOp, i32, cond,
        /*thenBuilder=*/
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc, const0);
        },
        /*elseBuilder=*/
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc, negToZeroOp.input());
        });
    return success();
  }
};

static void populateStandaloneToScfPatterns(RewritePatternSet &patterns) {
  patterns.add<NegToZeroLowering>(patterns.getContext());
}

class StandaloneToScfPass : public StandaloneToScfBase<StandaloneToScfPass> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateStandaloneToScfPatterns(patterns);

    ConversionTarget target(getContext());
    target.addIllegalOp<NegToZeroOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<Pass> mlir::standalone::createStandaloneToScfPass() {
  return std::make_unique<StandaloneToScfPass>();
}

