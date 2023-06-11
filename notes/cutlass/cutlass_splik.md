#### CUTLASS Splik-K
##### 1、背景
CUTLASS GEMM运算时对result C matrix进行tiling，以threadblock tile为组织，将结果matrix C切分成`MXN`块，其中每个块的计算需要`problem.k/thradblock.k`次循环。`problem.k`过大时，`block`中循环占比提高，可以通过设置`splitk` 启动更多的额外的`splitk-1`个`block`来共同分担这块计算。但引入了`epilogue`阶段的`blocks`间的`sync`行为。这里主要介绍下`CUTLASS`如何在`blocks`间进行`sync`。
##### 2、实现

首先看下代码
```
Semaphore semaphore(params.semaphore + block_idx, thread_idx);

// If performing a reduction via split-K, fetch the initial synchronization
if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
    
    // Fetch the synchronization lock initially but do not block.
    semaphore.fetch();

    // Indicate which position in a serial reduction the output operator is currently updating
    output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
}

// Wait on the semaphore - this latency may have been covered by iterator construction
if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
    
    // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
    if (threadblock_tile_offset.k()) {
    iterator_C = iterator_D;
    }

    semaphore.wait(threadblock_tile_offset.k());

}

/*
    Epilogue Operator
*/

//
// Release the semaphore
//

if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
    
    int lock = 0;
    if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

    // The final threadblock resets the semaphore for subsequent grids.
    lock = 0;
    }
    else {
    // Otherwise, the semaphore is incremented
    lock = threadblock_tile_offset.k() + 1;
    }

    semaphore.release(lock);
}



```
