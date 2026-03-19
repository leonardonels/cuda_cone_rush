# CUDA Kernel Documentation and Pipeline Interaction

This document explains how each CUDA kernel in this package works and how kernels interact inside the runtime pipeline, using the exact buffers and call sequence from the code.

## 1) End-to-end pipeline overview

The processing flow in `ControllerNode::runPipeline(unsigned int inputSize)` is:

1. Input conversion to GPU float array (`x, y, z, intensity`) stride 4.
2. Optional filtering stage (custom passthrough CUDA kernel).
3. Optional segmentation stage (custom RANSAC + compaction CUDA kernels).
4. Optional clustering stage (voxelization + union-find + dimension filtering CUDA kernels).
5. Publish filtered/segmented point clouds and final cone markers.

Concrete buffers in this flow:
- `d_input`: current active point cloud on GPU (`thrust::device_vector<float>`).
- `d_output`: scratch/output GPU cloud for next stage.
- `h_input`: host-side staging buffer used only for publish copies and BARQ path.
- `size`: stage output point count written from device counters.

Data stays in VRAM between stages via `d_input.swap(d_output)`. Host copies are only done for:
- publishing filtered/segmented clouds,
- final clustering cone centers.

## 2) Input conversion kernel

### `convertPointCloud2Kernel` (in `src/utils/cuda_pointcloud_converter.cu`)

Purpose:
- Convert ROS PointCloud2 raw bytes into contiguous float layout on GPU.
- Output format is always 4 floats per point: `[x, y, z, intensity]`.

How it works:
- 2D launch (`blockIdx.x/y`, `threadIdx.x/y`) maps one thread to one point at row/column.
- Computes source pointer with `row_step` and `point_step`.
- Reads `x`, `y`, `z` bytes and stores to output.
- Writes intensity as `0.0f` in this converter path.

Exact launch in code:
- `threads = dim3(16, 16)`
- `blocks = dim3((width + 15)/16, (height + 15)/16)`
- Output index: `out_idx = (r * width + c) * 4`

Pipeline interaction:
- Used in `scanCallback` when `USE_CUDA_POINTCLOUD_CONVERTER` is enabled.
- Runs in its own converter stream and synchronizes at the end, so `d_input` is ready before `runPipeline`.

## 3) Filtering kernel

### `passthroughFilterKernel` (in `src/filtering/cuda_filtering.cu`)

Purpose:
- Remove points outside configured X/Y/Z ranges in one pass.

How it works:
- One thread per point.
- Checks active axis limits (`doX/doY/doZ`).
- If point is valid, uses `atomicAdd(d_count, 1)` to reserve output slot.
- Writes compacted point to output buffer.

Exact launch and counters:
- Launch config in `CudaFilter::filterPoints`:
	- `threads = passthroughFilterKernel_block_size`
	- `blocks = (inputSize + threads - 1) / threads`
- Device counter is `d_count` (`thrust::device_vector<unsigned int>` size 1).
- Counter reset each frame with `cudaMemsetAsync(raw_count, 0, sizeof(unsigned int), stream)`.
- Output count copied to host pointer `outputSize` with `cudaMemcpyAsync(..., cudaMemcpyDeviceToHost)`.

Pipeline interaction:
- Called by `CudaFilter::filterPoints` from `runPipeline` when filtering is enabled.
- Produces compacted output and point count entirely on GPU.
- Controller swaps `d_input` and `d_output` after completion so next stage consumes filtered points.
- Optional publish path then copies `size * 4 * sizeof(float)` from `d_input` to `h_input` on `copy_stream`.

## 4) Segmentation kernels (ground removal)

### `ransacPlaneKernel` (in `src/segmentation/cuda_segmentation.cu`)

Purpose:
- Execute many RANSAC plane hypotheses in parallel (one block = one iteration).

How it works:
- Randomly picks 3 points per iteration.
- Builds plane model `(a, b, c, d)` from cross product.
- Rejects degenerate triangles and points beyond max segmentation distance heuristic.
- Each thread counts inliers for a subset of points.
- Block reduction accumulates inlier count and stores model for that iteration.

Exact mapping and memory:
- Grid layout is `<<<max_iter, ransacPlaneKernel_block_size>>>`.
- One block equals one RANSAC hypothesis (`iter = blockIdx.x`).
- Per-iteration outputs:
	- `plane_inliers_counts[iter]` (int)
	- `plane_models[iter]` (`float4(a,b,c,d)`).
- Shared reduction buffer: `__shared__ int s_counts[1024]`.
- Invalid hypotheses write `-1` into `plane_inliers_counts`.

Pipeline interaction:
- First major kernel in segmentation stage.
- Results are consumed by GPU-side max search (`thrust::max_element`) to choose best plane.
- Best plane is copied device-to-device into `d_bestPlane[0]` for next kernel.

### `markInliersKernel` and `markInliersFromDeviceKernel`

Purpose:
- Build an index mask where points close to best plane are marked as inliers.

How they work:
- One thread per point computes point-to-plane distance.
- Writes `1` for inlier, `0` for outlier.
- `markInliersFromDeviceKernel` reads best plane directly from a device pointer to avoid host sync.

Current path in code:
- `markInliersFromDeviceKernel` is launched, not `markInliersKernel`.
- Input mask buffer is `d_index` (`int` per point).
- Launch config: `blocks = (nCount + threads - 1) / threads`.

Pipeline interaction:
- `markInliersFromDeviceKernel` is the variant used in current segmentation flow.
- Output mask is passed to compaction kernel.

### `compactInliersKernel`

Purpose:
- Compact non-ground points into output cloud.

How it works:
- One thread per point checks mask.
- If point is not part of the segmented plane (`index != 1`), atomically appends to output.

Concrete behavior:
- In this implementation, segmentation removes plane inliers and keeps outliers.
- Device counter is `d_out_count[0]` (reset by `cudaMemsetAsync`).
- Output points are written directly to `out_points` passed by caller (`raw_out` from controller).

Pipeline interaction:
- Final segmentation kernel.
- Produces new compact cloud and output size used by controller.
- Controller swaps `d_input` and `d_output`, so clustering receives ground-removed cloud.

## 5) Clustering kernels

Clustering uses voxel-based grouping plus union-find connectivity, then geometric filtering of clusters.

### `boundingBoxKernel` (Kernel 1)

Purpose:
- Compute global min/max XYZ of input cloud on GPU.

How it works:
- Grid-stride loop accumulates local min/max per thread.
- Shared-memory reduction per block.
- Block results merged into global bbox with float atomic min/max helpers.

Exact layout and init:
- Global bbox array `d_bbox[6]` stores `[minX, minY, minZ, maxX, maxY, maxZ]`.
- Before launch, bbox is initialized from host with `[1e30, 1e30, 1e30, -1e30, -1e30, -1e30]`.
- Shared memory bytes at launch: `6 * blockDim.x * sizeof(float)`.

Pipeline interaction:
- Produces `d_bbox` consumed by voxel-key kernel.

### `computeVoxelKeysKernel` (Kernel 2)

Purpose:
- Map each point to a voxel hash key.

How it works:
- Loads bbox from device.
- Computes grid dimensions from voxel sizes.
- Converts point coordinates to integer voxel indices.
- Stores hash: `ix + iy*gridX + iz*gridX*gridY`.

Important implementation details:
- Thread 0 computes and stores `d_grid[0]=gridX`, `d_grid[1]=gridY`.
- Conversion uses floor-like rounding (`__float2int_rd`) for voxel index.
- Grid dimensions use round-up (`__float2int_ru`) plus `+1` margin.

Pipeline interaction:
- Output keys are sorted and reduced by Thrust to unique occupied voxels.
- `d_voxelKeys` is copied to `d_origKeys` before sort so point order can be labeled later.

### `unionFindKernel` (Kernel 3)

Purpose:
- Merge neighboring occupied voxels into connected components (26-neighborhood).

How it works:
- One thread per occupied voxel.
- Decodes voxel index from hash.
- Checks all 26 neighbors.
- Finds neighbor existence by binary search in sorted key list.
- Uses device union-find (`uf_find`, `uf_union`) with atomic CAS.

Key data structures:
- Input voxel set is `d_filteredKeys` (voxels passing count threshold).
- Parent array `d_parent` starts as identity via `thrust::sequence`.
- `uf_union` always links larger root under smaller root (deterministic tie policy).

Pipeline interaction:
- Builds parent structure for cluster identities.

### `flattenParentKernel` (Kernel 4)

Purpose:
- Path-compress union-find parent array to stable roots.

How it works:
- One thread per voxel calls `uf_find` and writes final root.

Pipeline interaction:
- Required before assigning labels to points.

### `assignClusterLabelsKernel` (Kernel 5)

Purpose:
- Assign each original point a cluster label.

How it works:
- One thread per point.
- Binary-search point voxel key in filtered voxel list.
- Label = flattened parent root, or `-1` if filtered out.

Specific buffers:
- Input key per point: `d_origKeys` (unsorted copy from step 2).
- Filtered key list: `d_filteredKeys`.
- Output labels: `d_pointLabels` (`int`, one per input point).

Pipeline interaction:
- Feeds per-point labels to per-cluster bbox accumulation.

### `clusterBBoxKernel` (Kernel 6)

Purpose:
- Compute bounding box and size for each cluster.

How it works:
- One thread per point.
- Resolves compact cluster id from root label.
- Updates cluster min/max with float atomics.
- Increments cluster size with atomic add.

How compact IDs are built before this kernel:
- Unique cluster roots are computed from `d_parent` into `d_uniqueLabels`.
- `d_labelMap` is created as sequential `0..numClusters-1`.
- Kernel binary-searches root label in `d_labelKeys`/`d_uniqueLabels` to resolve compact cluster id.

Pipeline interaction:
- Produces cluster geometry inputs for cone-shape filtering.

### `dimensionFilterKernel` (Kernel 7)

Purpose:
- Keep only clusters whose dimensions and height are cone-like.

How it works:
- One thread per cluster.
- Applies min/max cluster size checks.
- Computes `dx, dy, dz` from cluster bbox.
- Applies configured geometric limits and `maxHeight`.
- Valid clusters atomically append center point `(cx, cy, cz)` to output cone list.

Exact filter expression in code:
- Size gate: `minClusterSize <= sz <= maxClusterSize`.
- Height gate: `mnZ < maxHeight`.
- Dimension gates:
	- `clusterMinX < dx < clusterMaxX`
	- `clusterMinY < dy < clusterMaxY`
	- `clusterMinZ < dz < clusterMaxZ`
- Output is `d_conePoints` with a device-side counter `d_countsDevice[3]`.

Pipeline interaction:
- Final clustering kernel.
- Host copies only the final cone count and cone centers, then publishes marker points.

## 6) Stream and synchronization behavior

- Compute kernels in filtering/segmentation/clustering run on `compute_stream`.
- Point cloud publish copies use `copy_stream`.
- Current implementation synchronizes at explicit points, so stages are mostly serialized for correctness.
- Device buffer swapping (`d_input.swap(d_output)`) is the handoff mechanism between pipeline stages.

Where sync happens in code:
- Filtering: stream sync after output-size D->H copy and after compacted D->D copy.
- Segmentation: one sync after final out-count D->H copy.
- Clustering: sync after copying `numCones` D->H, then another sync after copying cone points D->H.
- Publish copies on `copy_stream` are synchronized before ROS publish call.

## 7) Stage handoff summary

1. Converter writes `d_input` (`float[4*N]`).
2. Filter reads `d_input`, writes compact cloud to `d_output`, returns `size` via `d_count`, then controller swaps vectors.
3. Segmentation reads new `d_input`, runs RANSAC + mask + compaction, writes to `d_output`, returns `size` via `d_out_count`, then controller swaps vectors.
4. Clustering reads final `d_input`, computes cone centers into `d_conePoints`, copies `numCones` and cone centers to host, fills `cones->points`, publishes marker.

This design minimizes host-device transfers and keeps heavy computation on GPU kernels.

## 8) Exact call order per frame

The following order is what runs for one frame when all stages are enabled:

1. `convertPointCloud2Kernel` (if ROS path with CUDA converter enabled).
2. `passthroughFilterKernel`.
3. `ransacPlaneKernel`.
4. `markInliersFromDeviceKernel`.
5. `compactInliersKernel`.
6. `boundingBoxKernel`.
7. `computeVoxelKeysKernel`.
8. Thrust ops: sort, reduce_by_key, copy_if, unique.
9. `unionFindKernel`.
10. `flattenParentKernel`.
11. `assignClusterLabelsKernel`.
12. `clusterBBoxKernel`.
13. `dimensionFilterKernel`.
14. D->H copy of cone count and cone centers, then ROS marker publish.
