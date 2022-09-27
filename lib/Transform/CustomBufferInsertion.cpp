//===- CustomBufferInsertion.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-stream/Transform/CustomBufferInsertion.h"
#include "circt-stream/Transform/Passes.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetOperations.h"

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace circt_stream;

namespace circt_stream {
#define GEN_PASS_DEF_CUSTOMBUFFERINSERTION
#include "circt-stream/Transform/Passes.h.inc"
} // namespace circt_stream

static bool doesLoop(Operation *op) {
  SmallVector<Operation *> stack = llvm::to_vector(op->getUsers());
  DenseSet<Operation *> visited;

  while (!stack.empty()) {
    Operation *curr = stack.pop_back_val();
    if (visited.contains(curr))
      continue;
    visited.insert(curr);

    if (curr == op)
      return true;

    llvm::copy(curr->getUsers(), std::back_inserter(stack));
  }
  return false;
}

template <typename Func>
static void traverse(Operation *op, DenseSet<Operation *> &res, Func f) {
  SmallVector<Operation *> stack = {op};

  while (!stack.empty()) {
    auto *curr = stack.pop_back_val();
    if (res.contains(curr))
      continue;
    res.insert(curr);

    f(curr, stack);
  }
}

static void insertBuffer(Location loc, Value operand, OpBuilder &builder,
                         unsigned numSlots, BufferTypeEnum bufferType) {
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointAfterValue(operand);
  auto bufferOp = builder.create<handshake::BufferOp>(
      loc, operand.getType(), numSlots, operand, bufferType);
  operand.replaceUsesWithIf(
      bufferOp, function_ref<bool(OpOperand &)>([](OpOperand &operand) -> bool {
        return !isa<handshake::BufferOp>(operand.getOwner());
      }));
  builder.restoreInsertionPoint(ip);
}

static LogicalResult findAllLoopElements(Operation *op,
                                         DenseSet<Value> &elements) {

  DenseSet<Operation *> preds, succs;

  traverse(op, preds,
           [&](Operation *curr, SmallVectorImpl<Operation *> &stack) {
             for (auto operand : curr->getOperands()) {
               if (operand.isa<BlockArgument>())
                 continue;
               stack.push_back(operand.getDefiningOp());
             }
           });

  traverse(op, succs,
           [&](Operation *curr, SmallVectorImpl<Operation *> &stack) {
             for (auto *user : curr->getUsers()) {
               stack.push_back(user);
             }
           });

  llvm::set_intersect(preds, succs);
  for (auto *o : preds) {
    for (auto res : o->getResults()) {
      assert(res.hasOneUse());
      bool valInCycle = llvm::any_of(res.getUsers(), [&](Operation *user) {
        return preds.contains(user);
      });
      if (valInCycle)
        elements.insert(res);
    }
  }

  return success();
}

static LogicalResult findCycleElements(Region &r,
                                       DenseSet<Value> &cycleElements) {
  SmallVector<Operation *> loopStarts;
  Block *b = &r.front();
  llvm::copy_if(b->getOps<BufferOp>(), std::back_inserter(loopStarts),
                doesLoop);

  for (auto op : loopStarts)
    if (failed(findAllLoopElements(op, cycleElements)))
      return failure();

  return success();
}

static bool isUnbufferedChannel(Value &val) {
  assert(val.hasOneUse());
  Operation *definingOp = val.getDefiningOp();
  Operation *usingOp = *val.user_begin();
  return !isa_and_nonnull<BufferOp>(definingOp) && !isa<BufferOp>(usingOp);
}

static LogicalResult customRegionBuffer(Region &r) {
  DenseSet<Value> cycleElements;
  if (failed(findCycleElements(r, cycleElements)))
    return failure();

  OpBuilder builder(r.getParentOp());
  if (cycleElements.empty())
    return bufferRegion(r, builder, "all", 1);

  for (auto &defOp : llvm::make_early_inc_range(r.getOps())) {
    for (auto res : defOp.getResults()) {
      if (cycleElements.contains(res) || !isUnbufferedChannel(res))
        continue;
      insertBuffer(res.getLoc(), res, builder, 1, BufferTypeEnum::seq);
    }
  }

  return success();
}

namespace {
class CustomBufferInsertionPass
    : public circt_stream::impl::CustomBufferInsertionBase<
          CustomBufferInsertionPass> {
public:
  void runOnOperation() override {
    // Assumption: only very small cycles and no memory operations
    auto f = getOperation();
    if (f.isExternal())
      return;

    if (failed(customRegionBuffer(f.getBody())))
      signalPassFailure();
  }
};
} // namespace
