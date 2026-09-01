# Changelog

All notable changes to this crate are documented in this file.
This file is auto-generated from the workspace CHANGELOG.md by scripts/split-changelog.py.

## 0.5.7 - 2026-09-01

#### Added

- *(vector)* drain writes into an index while it builds

---

## 0.5.6 - 2026-09-01

#### Testing

- *(vector)* wait for waiters to queue instead of sleeping

---

## 0.5.2 - 2026-08-30

#### Fixed

- *(ci)* teach the changelog splitter the current heading layout

---

## 0.5.1 - 2026-08-29

#### Added

- *(vector)* opt-in cache-locality reorder on bulk build
- *(vector)* apply BFS reorder permutation across HNSW stores
- *(vector)* BFS visit-order permutation for HNSW cache-locality reorder
- *(vector)* add the centroid shard router (closure replication + adaptive fan-out)
- *(vector)* serving health + HLC freshness watermark for indexes
- *(vector)* distance-kernel call counter behind a feature
- *(vector)* brute-force cluster assignment for bulk_build
- *(vector)* seed leader sample in bulk_build
- *(vector)* bulk_build skeleton with insert_batch fallback
- *(vector)* maxsim scoring kernel for late-interaction
- *(vector)* add SearchMode::Exact for recall=1.0 brute-force kNN
- *(vector)* add f32-only contiguous layer-0 block
- *(vector)* add reusable SearchScratch buffers + pool
- *(vector)* mirror RaBitQ code and scalars to InlineLayer0
- *(vector)* InlineLayer0 reserves RaBitQ scalars + bit-aware code
- *(vector)* mirror layer-0 neighbours to InlineLayer0 on every write
- *(vector)* mirror f32 vector and label to InlineLayer0 on insert
- *(vector)* InlineLayer0 store for HNSW layer-0 contiguous payload
- *(vector)* RaBitQ EndOfSearch oversampling knob
- *(vector)* RerankMode knob (Inline | EndOfSearch | None) for RaBitQ
- *(vector)* R863 RobustPrune α-pruning neighbour selector
- *(vector)* K-cluster IVF via K-means Lloyd, default K=16 for RaBitQ
- *(vector)* K=1 IVF centering for RaBitQ cosine reconstruction
- *(vector)* RaBitQ paper Equation 20 asymmetric kernel (4-bit query)
- *(vector)* RaBitQ supports any dim via internal padding to next mult of 64
- *(vector)* wire Extended-RaBitQ 2/3/4-bit through HnswIndex
- *(vector)* Extended-RaBitQ 2/3/4-bit codec primitive
- *(vector)* LsmVectorTier - production binding of VectorTierStorage
- *(vector)* VectorTierStorage trait + write hooks in HnswIndex (ADR-033)
- *(vector)* set_rabitq_params for segment reload + serde round-trip tests
- *(vector)* wire RaBitQ codec into HNSW search hot path
- *(vector)* RaBitQ codec foundation + popcount distance kernel
- *(vector)* AtomicU64-packed entry-point + CAS-loop promotion
- *(vector)* C3 day 4 - prune-pass restores recall, insert_batch wired to parallel apply
- *(vector)* C3 day 3 - apply_insert_plans_parallel (opt-in, lossy)
- *(vector)* C3 day 2 - &self write helpers + cas_add_neighbour_to
- *(vector)* C3 day 1 - cas_append + replace primitives
- *(vector)* C2 day 3 - insert_batch vs serial criterion bench
- *(vector)* C2 day 2 - insert_batch with rayon parallel planning
- *(vector)* C2 day 1 - split insert into compute_insert_plan + apply
- *(vector)* C1 day 7 - parallel-search QPS bench
- *(vector)* C1 day 6 - HnswConfig::max_elements drives pre-allocation
- *(vector)* C1 day 5 - atomic mirror is the sole storage
- *(vector)* C1 day 4 - granular dual-write helpers for atomic mirror
- *(vector)* C1 day 3b - search read path now lock-free
- *(vector)* C1 day 3a - auto-sync atomic mirror after every insert/update
- *(vector)* C1 day 2 - atomic-mirror field + manual sync helper
- *(vector)* IndexHealthState + HnswBuildScheduler (rebalance prereqs)
- *(vector)* C1 day 1 - AtomicNeighbourList<N> scaffold for lock-free HNSW
- *(bench)* R700+R704 - coordinode-bench harness + ann-benchmarks SIFT1M adapter (Stage 1)
- CoordiNode v0.1.0-alpha.1 - graph + vector + full-text engine

#### Documentation

- *(vector)* bulk_build path and follow-up boundary
- *(vector)* C3 day 6 - record measured 14.6× speedup in bench doc

#### Fixed

- *(vector)* silence unused_unsafe in distance kernel bench
- *(vector)* require results full before terminating HNSW search
- *(vector)* reconstruct ‖x‖ from IVF code header for cosine rerank
- *(vector)* track f32 distance in HNSW results-heap, RaBitQ in frontier
- *(vector)* RaBitQ per-vector correction scaling (chroma-style)
- *(vector)* build HNSW on exact f32, only use RaBitQ at search time
- *(vector)* apply-phase backfill keeps closer candidates over incumbents
- *(vector)* HNSW search beam must be at least k for any caller
- *(modality,vector)* pass &StorageEngine in doctests + add LockFreeNeighbours::is_empty
- *(vector)* preserve back-edges when neighbour list is at M_MAX0 cap
- *(vector)* dedupe duplicate ids within insert_batch + proptest stress
- *(vector)* update HNSW graph position when node vector is overwritten (G082)

#### Performance

- *(vector)* batch vector-tier reads through engine multi_get
- *(vector)* prune HNSW back-edges with the diversity heuristic
- *(vector)* size the layer-0 block to the effective neighbour degree
- *(vector)* co-locate neighbours and f32 in one contiguous block, drop SoA neighbour list
- *(vector)* free f32 from contiguous blocks on offload, drop redundant SoA store
- *(vector)* default RobustPrune alpha to 1.15 for cosine builds
- *(vector)* walk inline layer-0 neighbour rows in place during search
- *(vector)* stop eager RaBitQ code indexing on unquantized visits
- *(vector)* cache inverse node norms, multiply instead of divide in cosine
- *(vector)* prefetch next frontier candidate's neighbour ids
- *(vector)* skip per-node RaBitQ lookup in prefetch when codec inactive
- *(vector)* prefetch the full vector span, not one cache line
- *(vector)* bind SIMD kernel pointer once on x86_64
- *(vector)* skip exact re-distance when no quantizer is active
- *(vector)* both-norms fast path for insert pruning distances
- *(vector)* pre-normalise cosine vectors at insert to drop divide
- *(vector)* cache per-node L2 norm to skip per-visit norm pass
- *(vector)* skip RaBitQ+SQ8 chain when neither active
- *(vector)* prefetch top candidate vector after push
- *(metrics)* mark distance kernel implementations inline
- *(vector)* thread-local VisitedPool storage
- *(vector)* wire data_level0 into search prefetch + f32 read
- *(vector)* prefer SoA over inline for f32 vector reads
- *(vector)* read RaBitQ code from InlineLayer0 in cosine search
- *(vector)* read f32 vector from InlineLayer0 in compute_exact_distance
- *(vector)* read layer-0 neighbours from InlineLayer0 in search
- *(vector)* hoist nodes.len() out of search inner loop
- *(vector)* unchecked visited.check_and_mark on search hot path
- *(vector)* SIMD-ify FHT butterfly via AVX2 / NEON with runtime detect
- *(vector)* inline RaBitQQuery bit-planes via SmallVec
- *(vector)* skip prefetch_node_vector entirely on cosine + RaBitQ
- *(vector)* flat rabitq_code_ptr cache eliminates SoA load on prefetch
- *(vector)* split layer-0 neighbours flat + skip f32 prefetch on RaBitQ
- *(vector)* inline RaBitQ code u64 words via SmallVec (D ≤ 256 = no heap)
- *(vector)* prefetch next neighbour's visited counter byte (hnswlib pattern)
- *(vector)* replace results-heap push+pop with peek_mut+swap
- *(vector)* prefetch RaBitQ code in addition to f32 vector
- *(vector)* fused 4-plane AND+popcount kernel for RaBitQ asymmetric path
- *(vector)* cache node norm in cosine rerank to drop per-call norm pass
- *(vector)* replace Gram-Schmidt rotation with FHT-Kac (O(D²)→O(D log D))
- *(vector)* flat contiguous vector store for HNSW distance hot path
- *(vector)* prefetch full vector range (8 cache lines @ d=128)
- *(vector)* pack Candidate/FarCandidate to 8 bytes (u32 idx)
- *(vector)* hoist alloc + cache farthest in HNSW search inner loop
- *(vector)* inline distance dispatch chain end-to-end
- *(vector)* store internal indices in HNSW neighbour lists
- *(vector)* revert x86_64 AVX2 L2/dot/L1 to single-acc FMA shape
- *(vector)* match hnswlib AVX2/512 kernel shape (single-acc, mul+add)
- *(vector)* multi-accumulator SIMD distance kernels + runtime AVX-512
- *(vector)* bit-pack RaBitQExtCode to paper-quoted sizes
- *(vector)* criterion harness for RaBitQ popcount kernel
- *(vector)* SQ8 dequantize into reusable scratch + SIMD
- *(vector)* cache query L2 norm per HNSW search (cosine path)
- *(vector)* C3 day 5b - parallel prune-pass via rayon
- *(vector)* C3 day 5a - dedupe backfill before prune-pass

#### Refactored

- extract unit tests into sibling files (query, storage, vector, search)
- extract unit tests into sibling test files
- *(vector)* SoA split of HnswNode payload arrays
- *(vector)* drop intermediate quantized disk tier (ADR-033 final)
- *(vector)* migrate quantization config from bool to QuantizationCodec enum

#### Testing

- *(vector)* stabilize the reordered bulk-build self-recall check
- *(hnsw)* assert batch recall vs ground truth, not serial topology
- *(vector)* bulk_build vs insert_batch criterion arm
- *(vector)* RaBitQ cosine dim=100 reproducer narrows bug to scale
- *(vector)* isolate RaBitQ recall bug + cap rayon to 4 threads in CI
- *(vector)* end-to-end RaBitQ + LSM tier wiring
- *(vector)* regression tests for HNSW recall when ef_search < k
- *(vector)* wire loom interleaving suite for AtomicNeighbourList
- *(vector)* stress AtomicNeighbourList cas_append vs concurrent snapshot
- *(vector)* add proptest stress for multi-batch + concurrent search

#### Revert

- move per-label vector shard routing out of CE
- *(vector)* undo "flat contiguous vector store" - bench regressed
- *(vector)* undo "prefetch full vector range" - bench regressed

---

## 0.5.0 - 2026-06-27

#### Added

- *(vector)* opt-in cache-locality reorder on bulk build
- *(vector)* apply BFS reorder permutation across HNSW stores
- *(vector)* BFS visit-order permutation for HNSW cache-locality reorder
- *(vector)* add the centroid shard router (closure replication + adaptive fan-out)
- *(vector)* serving health + HLC freshness watermark for indexes
- *(vector)* distance-kernel call counter behind a feature
- *(vector)* brute-force cluster assignment for bulk_build
- *(vector)* seed leader sample in bulk_build
- *(vector)* bulk_build skeleton with insert_batch fallback
- *(vector)* maxsim scoring kernel for late-interaction
- *(vector)* add SearchMode::Exact for recall=1.0 brute-force kNN
- *(vector)* add f32-only contiguous layer-0 block
- *(vector)* add reusable SearchScratch buffers + pool
- *(vector)* mirror RaBitQ code and scalars to InlineLayer0
- *(vector)* InlineLayer0 reserves RaBitQ scalars + bit-aware code
- *(vector)* mirror layer-0 neighbours to InlineLayer0 on every write
- *(vector)* mirror f32 vector and label to InlineLayer0 on insert
- *(vector)* InlineLayer0 store for HNSW layer-0 contiguous payload
- *(vector)* RaBitQ EndOfSearch oversampling knob
- *(vector)* RerankMode knob (Inline | EndOfSearch | None) for RaBitQ
- *(vector)* R863 RobustPrune α-pruning neighbour selector
- *(vector)* K-cluster IVF via K-means Lloyd, default K=16 for RaBitQ
- *(vector)* K=1 IVF centering for RaBitQ cosine reconstruction
- *(vector)* RaBitQ paper Equation 20 asymmetric kernel (4-bit query)
- *(vector)* RaBitQ supports any dim via internal padding to next mult of 64
- *(vector)* wire Extended-RaBitQ 2/3/4-bit through HnswIndex
- *(vector)* Extended-RaBitQ 2/3/4-bit codec primitive
- *(vector)* LsmVectorTier - production binding of VectorTierStorage
- *(vector)* VectorTierStorage trait + write hooks in HnswIndex (ADR-033)
- *(vector)* set_rabitq_params for segment reload + serde round-trip tests
- *(vector)* wire RaBitQ codec into HNSW search hot path
- *(vector)* RaBitQ codec foundation + popcount distance kernel
- *(vector)* AtomicU64-packed entry-point + CAS-loop promotion
- *(vector)* C3 day 4 - prune-pass restores recall, insert_batch wired to parallel apply
- *(vector)* C3 day 3 - apply_insert_plans_parallel (opt-in, lossy)
- *(vector)* C3 day 2 - &self write helpers + cas_add_neighbour_to
- *(vector)* C3 day 1 - cas_append + replace primitives
- *(vector)* C2 day 3 - insert_batch vs serial criterion bench
- *(vector)* C2 day 2 - insert_batch with rayon parallel planning
- *(vector)* C2 day 1 - split insert into compute_insert_plan + apply
- *(vector)* C1 day 7 - parallel-search QPS bench
- *(vector)* C1 day 6 - HnswConfig::max_elements drives pre-allocation
- *(vector)* C1 day 5 - atomic mirror is the sole storage
- *(vector)* C1 day 4 - granular dual-write helpers for atomic mirror
- *(vector)* C1 day 3b - search read path now lock-free
- *(vector)* C1 day 3a - auto-sync atomic mirror after every insert/update
- *(vector)* C1 day 2 - atomic-mirror field + manual sync helper
- *(vector)* IndexHealthState + HnswBuildScheduler (rebalance prereqs)
- *(vector)* C1 day 1 - AtomicNeighbourList<N> scaffold for lock-free HNSW
- *(bench)* R700+R704 - coordinode-bench harness + ann-benchmarks SIFT1M adapter (Stage 1)
- CoordiNode v0.1.0-alpha.1 - graph + vector + full-text engine

#### Documentation

- *(vector)* bulk_build path and follow-up boundary
- *(vector)* C3 day 6 - record measured 14.6× speedup in bench doc

#### Fixed

- *(vector)* silence unused_unsafe in distance kernel bench
- *(vector)* require results full before terminating HNSW search
- *(vector)* reconstruct ‖x‖ from IVF code header for cosine rerank
- *(vector)* track f32 distance in HNSW results-heap, RaBitQ in frontier
- *(vector)* RaBitQ per-vector correction scaling (chroma-style)
- *(vector)* build HNSW on exact f32, only use RaBitQ at search time
- *(vector)* apply-phase backfill keeps closer candidates over incumbents
- *(vector)* HNSW search beam must be at least k for any caller
- *(modality,vector)* pass &StorageEngine in doctests + add LockFreeNeighbours::is_empty
- *(vector)* preserve back-edges when neighbour list is at M_MAX0 cap
- *(vector)* dedupe duplicate ids within insert_batch + proptest stress
- *(vector)* update HNSW graph position when node vector is overwritten (G082)

#### Performance

- *(vector)* prune HNSW back-edges with the diversity heuristic
- *(vector)* size the layer-0 block to the effective neighbour degree
- *(vector)* co-locate neighbours and f32 in one contiguous block, drop SoA neighbour list
- *(vector)* free f32 from contiguous blocks on offload, drop redundant SoA store
- *(vector)* default RobustPrune alpha to 1.15 for cosine builds
- *(vector)* walk inline layer-0 neighbour rows in place during search
- *(vector)* stop eager RaBitQ code indexing on unquantized visits
- *(vector)* cache inverse node norms, multiply instead of divide in cosine
- *(vector)* prefetch next frontier candidate's neighbour ids
- *(vector)* skip per-node RaBitQ lookup in prefetch when codec inactive
- *(vector)* prefetch the full vector span, not one cache line
- *(vector)* bind SIMD kernel pointer once on x86_64
- *(vector)* skip exact re-distance when no quantizer is active
- *(vector)* both-norms fast path for insert pruning distances
- *(vector)* pre-normalise cosine vectors at insert to drop divide
- *(vector)* cache per-node L2 norm to skip per-visit norm pass
- *(vector)* skip RaBitQ+SQ8 chain when neither active
- *(vector)* prefetch top candidate vector after push
- *(metrics)* mark distance kernel implementations inline
- *(vector)* thread-local VisitedPool storage
- *(vector)* wire data_level0 into search prefetch + f32 read
- *(vector)* prefer SoA over inline for f32 vector reads
- *(vector)* read RaBitQ code from InlineLayer0 in cosine search
- *(vector)* read f32 vector from InlineLayer0 in compute_exact_distance
- *(vector)* read layer-0 neighbours from InlineLayer0 in search
- *(vector)* hoist nodes.len() out of search inner loop
- *(vector)* unchecked visited.check_and_mark on search hot path
- *(vector)* SIMD-ify FHT butterfly via AVX2 / NEON with runtime detect
- *(vector)* inline RaBitQQuery bit-planes via SmallVec
- *(vector)* skip prefetch_node_vector entirely on cosine + RaBitQ
- *(vector)* flat rabitq_code_ptr cache eliminates SoA load on prefetch
- *(vector)* split layer-0 neighbours flat + skip f32 prefetch on RaBitQ
- *(vector)* inline RaBitQ code u64 words via SmallVec (D ≤ 256 = no heap)
- *(vector)* prefetch next neighbour's visited counter byte (hnswlib pattern)
- *(vector)* replace results-heap push+pop with peek_mut+swap
- *(vector)* prefetch RaBitQ code in addition to f32 vector
- *(vector)* fused 4-plane AND+popcount kernel for RaBitQ asymmetric path
- *(vector)* cache node norm in cosine rerank to drop per-call norm pass
- *(vector)* replace Gram-Schmidt rotation with FHT-Kac (O(D²)→O(D log D))
- *(vector)* flat contiguous vector store for HNSW distance hot path
- *(vector)* prefetch full vector range (8 cache lines @ d=128)
- *(vector)* pack Candidate/FarCandidate to 8 bytes (u32 idx)
- *(vector)* hoist alloc + cache farthest in HNSW search inner loop
- *(vector)* inline distance dispatch chain end-to-end
- *(vector)* store internal indices in HNSW neighbour lists
- *(vector)* revert x86_64 AVX2 L2/dot/L1 to single-acc FMA shape
- *(vector)* match hnswlib AVX2/512 kernel shape (single-acc, mul+add)
- *(vector)* multi-accumulator SIMD distance kernels + runtime AVX-512
- *(vector)* bit-pack RaBitQExtCode to paper-quoted sizes
- *(vector)* criterion harness for RaBitQ popcount kernel
- *(vector)* SQ8 dequantize into reusable scratch + SIMD
- *(vector)* cache query L2 norm per HNSW search (cosine path)
- *(vector)* C3 day 5b - parallel prune-pass via rayon
- *(vector)* C3 day 5a - dedupe backfill before prune-pass

#### Refactored

- extract unit tests into sibling files (query, storage, vector, search)
- extract unit tests into sibling test files
- *(vector)* SoA split of HnswNode payload arrays
- *(vector)* drop intermediate quantized disk tier (ADR-033 final)
- *(vector)* migrate quantization config from bool to QuantizationCodec enum

#### Testing

- *(vector)* stabilize the reordered bulk-build self-recall check
- *(hnsw)* assert batch recall vs ground truth, not serial topology
- *(vector)* bulk_build vs insert_batch criterion arm
- *(vector)* RaBitQ cosine dim=100 reproducer narrows bug to scale
- *(vector)* isolate RaBitQ recall bug + cap rayon to 4 threads in CI
- *(vector)* end-to-end RaBitQ + LSM tier wiring
- *(vector)* regression tests for HNSW recall when ef_search < k
- *(vector)* wire loom interleaving suite for AtomicNeighbourList
- *(vector)* stress AtomicNeighbourList cas_append vs concurrent snapshot
- *(vector)* add proptest stress for multi-batch + concurrent search

#### Revert

- move per-label vector shard routing out of CE
- *(vector)* undo "flat contiguous vector store" - bench regressed
- *(vector)* undo "prefetch full vector range" - bench regressed

---

## 0.3.11 - 2026-04-14

#### Fixed

- *(vector)* update HNSW graph position when node vector is overwritten (G082)
