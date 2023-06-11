#### CUTLASS Splik-K
##### 1、背景
CUTLASS GEMM运算时对result C matrix进行tiling，以threadblock tile为组织，将结果matrix C切分成`MXN`块，其中每个块的计算需要`problem.k/thradblock.k`次循环。`problem.k`过大时，`block`中循环占比提高，可以通过设置`splitk` 启动更多的额外的`splitk-1`个`block`来共同分担这块计算。但引入了`epilogue`阶段的`blocks`间的`sync`行为。这里主要介绍下`CUTLASS`如何在`blocks`间进行`sync`。
##### 2、实现

首先看下代码。

- 第一步，使用`block_idx`初始化`semphore`，`block_idx`也就是result matrix中的tile idx。注意下这里的m维度是增长最快的维度，n是外层维度。
- 第二步，调用`semphore`的`fetch`方法，去load gpu global mem上的一个地址到reg上，通过嵌入ptxcode。
  - 在Amper架构下，调用的为`ld.global.acquire.gpu.b32 reg addr`，这里注意下只有当前block内的`thread id = 0`才会去访问这块。其实就是不同的`blocks`间做个`sync`，每个block中只会有一个thread会尝试执行上面这个动作。去load global上的一个address value，类似做了一个flag去对不同`blocks`做了一个`sync`。
  - 要同步的`blocks`数目为`splitk`。
- 第三步，设置下output_op的partioins，传入的参数为当前block tile的offset，以及grid tile shape，可以简单理解为当前block在blocks中的idx以及blocks的数目。
  - output_op为epilgoue/threadblock目录下的class，其`set_k_partioin`方法会将待同步的`blocks`中idx不为0的blocks的`beta_`设置为1（也就是epilgoue中不对bias做scale）。除此之外，对于待同步的block中非最后一个，将他们的的threadshold设置为-1（这个动作对于不同的epilgoue不同，这里以为+bias+relu为例）。
- 第四步，`blocks`中拿到global mem地址访问的那个会阻塞其他blocks。此时，做个判断，如果blocks中非，第一个则iterator_C指向iterator_D（也就是，bias iterator指向 destination iterator，cutlass中code的注释写到，对于非首个blocks，source matrix已经在D tensor中了）。
  - 随后，`semaphore`循环等待当前block idx索引相关信号量获取（fetch成功）。
- 第五步，执行epilgoue op。
- 第六步，释放semaphore。当第一个block idx执行完epilgoue后，后续的都不需要再进行bias操作了。后续的block只需要只scale * acc + beta(1) * destination。所以epilogue op的set_k_partition那块才会做set(beta, 1)的行为。
  - 释放时，blocks中第一个block的`semahore`执行释放`release(1)`，底层是调用`ptx`代码`asm volatile ("st.global.release.gpu.b32 [%0], %1;\n" : : "l"(lock), "r"(status));`。
  - 随后，其余的blocks此时应该停留在上述的第四步中的`fetch`上，此时会依次串行执行，重复后4-6这几步，直到所有的blocks做完。

##### 额外
- 1、每个block中只有第一个thread会负责做也就是thread_idx==0是才负责同步。
- 2、可以这么理解block_0，做完bias的动作，后续的block_x只需要负责吧mma的结果add上去即可。
- 3、如果想要在一个epilogue中做一些骚操作，例如两个gemm结果的elementwise的mul，只能放到两个epilgoue。

##### Two Gemm

block0: ()

```cpp
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


