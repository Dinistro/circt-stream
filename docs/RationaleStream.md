# Stream Dialect Rationale

This document describes various design points of the `stream` dialect.
This follows in the spirit of other 
[MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

Streaming libraries and abstractions become more popular as languages and tools adopt such concepts.
Furthermore, some high-level synthesis tools provide library support for streaming code, which then can be lowered to hardware.
In the spirit of MLIR and CIRCT, we anticipate that providing higher-level
abstraction in the form of a dialect simplifies further implementation efforts by providing a uniform interface.

## Types

The `stream` dialect introduces a single type that defines a stream by its element types. An element can either be an integer or a tuple of element types.

Examples:

```
!stream.stream<i64>
!stream.stream<tuple<i32, tuple<i8, i64>>>
```

## Operations

There are two different kinds of operations:
1. A set of operations that work directly with streams. These operations all consume and produce a variable amount of streams.
2. Auxiliary operations that help to work with elements of the stream, e.g., packing or unpacking tuples, yielding elements, etc.

So far, the `stream` dialect supports the following set of stream operations: `map`, `filter`, `reduce`, and `create`. 
The first three expect regions that define the computation to be performed on each stream element. Note that the region arguments differ depending on the operation and the element types of the streams passed in.  

Example:

```mlir
  %res = stream.map(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
  ^0(%val : i32):
    %0 = arith.constant 1 : i32
    %r = arith.addi %0, %val : i32
    stream.yield %r : i32
  }
}
```

## Lowering

One natural target for the streaming abstraction to lower is the handshake dialect. 
The handshake dialect is somewhat stable, and the `StandardToHandshake` pass can be reused to lower the regions of the operations. 

The streaming abstraction can be lowered to a task pipelined handshake representation. 
Each stream becomes an handshaked value of the element type, and all the operations defined on this stream are applied directly to these values. 
Note that certain operations like `filter` and `reduce` can terminate incoming control flow.

### End-of-Stream signal

Some operations, e.g., `reduce`, only produce a result when the incoming stream terminates. 
To allow such behavior upon lowering each stream provides an `EOS` signal which is asserted once
the stream is ending.

