## Chapter 3 — Multidimensional Grids and Data

### What I learned

**Thread execution model (the big one)**
Every loop you parallelize disappears as a `for` and becomes threads instead —
the GPU runs all iterations at once, one per thread. The thread's index
(`row = blockIdx.x * blockDim.x + threadIdx.x`) is its *identity*: which piece
of the output it owns. That's fixed for the whole kernel. Any inner `for` loops
are just indices the thread iterates over (the "pointer" it moves), not the
thread moving. Rule that unblocks everything: ask "what does one thread produce,
and how is that thing identified?" — that answer is your threadIdx variable.

**Multidimensional indexing**
Data is always stored flat in memory regardless of dimensionality. What changes
is how you compute the 1D index:
- 2D row-major:    `row * width + col`
- 2D column-major: `col * height + row`  (Fortran-style, the formula flips)
- 3D row-major:    `z * (width * height) + y * width + x`

Pattern: to skip one full dimension, multiply by the size of everything below it.
Each new dimension adds a term scaled by the size of all inner dimensions.

**Execution configuration**
- 1D launch (one thread per row/column/element): `<<<(N + 255) / 256, 256>>>`
- Use the integer ceiling `(N + size - 1) / size`, not `ceil(N/256.0)` — exact,
  faster, no float precision risk, no `<cmath>` dependency.
- Use `size_t` for byte/element counts on large arrays — `int` overflows
  silently (e.g. 45000² * 4 bytes blows past int and crashes with SIGSEGV).

**The guard is mandatory**
Whenever you use ceiling for block count, you launch more threads than you need
(N rarely divides evenly by block size). The `if (row < width && col < height)`
guard disables the extra threads so they don't write to invalid memory.
Example from ex.3: launched 48,640 threads, only 45,000 passed the guard —
the other 3,640 (the border) did nothing. That waste is normal and tiny.

**Memory coalescing (intro — full treatment in ch.6)**
When neighboring threads (0,1,2,3...) access *contiguous* memory addresses at
the same time, the hardware merges them into one transaction (fast). When they
access *strided* addresses (jumping by Width), it needs many separate
transactions (slow). This is why the one-thread-per-column kernel beats the
one-thread-per-row kernel: in column-major access the neighbors land on
contiguous addresses.

**Parallelism trade-off (ex.1c)**
One-thread-per-row and one-thread-per-column both launch only Width threads and
make each thread do a full serial loop — terrible parallelism. The original
one-thread-per-element kernel launches Width² threads and saturates the GPU.
Ranking: original 2D > per-column (coalesces) > per-row (strided).

### Exercises done
1. Matrix mul, one thread per row (a) / per column (b) + pros/cons (c) — coalescing analysis
2. Matrix-vector mul kernel + host stub (watch: B is V×V, C and A are just V — two different sizes)
3. Read a config and count: threads/block (512), grid threads (48,640), blocks (95), threads on line 05 (45,000)
4. 2D index: row-major (8,010) vs column-major (5,020)
5. 3D tensor index: 1,008,010

### Open questions → answered in Chapter 4
- How many threads does a GPU physically have? → SMs (logical vs physical threads)
- Does memory access take time on GPU? → no, it's slow; latency hiding disguises it
- Are the wasted guard threads expensive? → no; warp divergence / occupancy explain the real costs