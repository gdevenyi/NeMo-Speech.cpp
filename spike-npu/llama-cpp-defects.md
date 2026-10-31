# llama.cpp defects found during the NPU work

Date: 2026-10-30. Pin: `bd4f514`. Current `master` checked: `889f2ec`.

Line numbers are for the pin. "Also in `master`" means that the same code is there. I read that code but did not test each defect on `master`, except where this file says so.

## Summary

| # | Component | Problem | Effect | Also in `master` |
|---|---|---|---|---|
| 1 | `ggml-backend.cpp` | Scheduler rewrites node sources in place | A reused graph with a split reads a stale copy | Yes, line 1803 |
| 2 | `ggml-openvino` | Static NPU path requires `inp_pos` | Non-LLM graphs throw | Yes |
| 3 | `ggml-openvino` | VIEW does not slice if strides differ on axes of size 1 | Wrong shape, translation throws | Yes, tested |
| 4 | `ggml-openvino` | Unused RESHAPE or VIEW is not an output | Result is lost; NPU compile fails | Yes |
| 5 | `ggml-openvino` or NPU compiler | Zero-point weight form loses accuracy on the NPU (bias form does not) | 8-14% error at one `MUL_MAT`; transcript errors | Same setting at line 1589; accuracy not tested |
| 6 | `ggml-openvino` | `supports_op` accepts graphs that the translator rejects | No CPU fallback | Not checked |
| 7 | `ggml-openvino` | Reshape `op_case` heuristic misreads shapes | Translation throws | Fixed for the generic path |

## 1. Scheduler rewrites node sources in place

This one may be a usage contract and not a bug. llama.cpp builds a new graph for each decode, so it does not reuse a graph. NeMo-Speech.cpp caches graphs and reuses them.

`ggml_backend_sched_split_graph` replaces the source of a node in the caller's graph:

```cpp
// ggml/src/ggml-backend.cpp:1419
node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
```

`ggml_backend_sched_reset` does not restore it. So on the next run of the same graph:

1. The node's source is now the CPU copy, a leaf that is already on the CPU.
2. The scheduler finds no backend change and makes no copy.
3. The node reads the first run's values for every later run.

**Observed:** the graph has an argmax on the CPU after logits from OpenVINO. The logits change on each run (mean -1558, -790, -770), but the copy stays at the first run's value (mean -49.27), and every frame decodes to blank.

**Spike fix:** save `node->src` when the graph is new, and restore it before each `ggml_backend_sched_alloc_graph`. With this fix, the OpenVINO CPU device gives the same transcript as the ggml CPU backend. The fix is in the engine, not in ggml.

**Decision for you:** ask the ggml maintainers if reuse of a graph is supported. If it is not supported, a note in `ggml-backend.h` would help. The engine fix is necessary in both cases.

## 2. Static NPU path requires `inp_pos`

`ov_graph_compute_static` calls `get_inp_pos_tensor(cgraph)` at `utils.cpp:1164`, and that function throws if the graph has no such tensor (`utils.cpp:1751`). An ASR encoder graph has no `inp_pos`, so it fails with:

```text
get_inp_pos_tensor: inp_pos not found in cgraph
```

The whisper.cpp support in #27843 adds KV-cache cases, but I found no route for an encoder-only graph on the NPU. I did not run whisper.cpp to confirm this.

The documentation says that the backend supports "text-only models", so this may be a known limit and not a defect.

**Spike fix:** `is_naive()` returns true for a graph with no `inp_pos`. `naive_compute` then needs a compile cache, because it compiles on each call. The spike keys the cache on a hash of nodes, data pointers, ops and shapes.

## 3. VIEW does not slice if strides differ on axes of size 1

`openvino/op/view.cpp:178` requires equal strides on all axes. `ggml_view_1d` gives a different stride on an axis of size 1, where the stride has no effect. The translator then passes the full tensor through.

**Example:** `GET_ROWS` gives [57344,1,2,1], and a view selects [57344,1,1,1]. The next reshape fails:

```text
Shape inference input shapes {[1,2,1,57344],[4]}
Requested output shape [1,1,56,1024] is incompatible with input shape
```

**Spike fix:** compare strides only on axes where the output size is above 1. The spike also sets `m_is_static = true` in the naive decoder, so that this generic slice runs at all.

## 4. Unused RESHAPE or VIEW is not an output

`GgmlOvDecoder::compute_model_outputs` skips these ops at `ggml-decoder.cpp:1211`:

```cpp
if (cur_node->op == GGML_OP_NONE || cur_node->op == GGML_OP_VIEW || cur_node->op == GGML_OP_RESHAPE) {
    continue;
}
```

If the last node of a graph is a RESHAPE, the graph can have no outputs. The RNNT joint graph ends this way (`MUL_MAT`, `ADD`, `RESHAPE`).

**Effect on the CPU device:** the model compiles, but the result is not written back.

**Effect on the NPU:** the compile fails, with a value that changes on each run (0, 874, 4114557787):

```text
The number of input types 4 does not match the number of function arguments 874
```

The changing value suggests that the NPU compiler reads uninitialized data for a model with no results. That part may deserve a separate report to the Intel NPU driver project. I did not confirm it.

**Spike fix:** add a RESHAPE or VIEW to the outputs if no node in the graph uses it. `use_counts` did not work for this test, so the spike counts users directly.

## 5. Quantized weight forms give different accuracy on the NPU

**This is not a defect in the bias setting. Do not change it.** An earlier version of this file called it a defect and gave `use_bias = false` as a fix. That edit caused transcript errors.

`create_weight_node` sets `use_bias = naive` at `ggml-decoder.cpp:1429`, so the generic path uses the bias form. The non-naive path uses the zero-point form. The two forms differ a lot on the NPU:

| Form | NPU chunk avg, 160 ms chunks | Error at the first large `MUL_MAT`, against OpenVINO CPU F32 | Transcript |
|---|---|---|---|
| Bias | 466 ms | 0.07-0.10% | Correct |
| Zero-point | 187 ms | 8-14%, peak 47-67 on values up to about 1170 | Errors |

The `MUL_MAT` here has 4352 inputs, 1024 outputs and `Q8_0` weights, and its inputs follow a ReLU.

This may matter to llama.cpp because its normal NPU path uses the zero-point form, and the documentation says that quantized accuracy validation is still in progress. I did not test an LLM. I also do not know why the form changes the accuracy. A faster, lower-precision integer path in the NPU compiler is my guess, and I have no direct evidence for it.

A useful report needs a small reproduction: one `MUL_MAT` with `Q8_0` weights, compared across the two forms on the NPU.

## 6. `supports_op` accepts graphs that the translator rejects

The encoder graph had one split, and the translator then threw on it. So `supports_op` accepted every node, and the scheduler had no chance to use the CPU. The documentation states this limit: "Unsupported ops or unsupported op shapes/cases fail during OpenVINO translation."

A fix is not simple, because the failures depend on the graph pattern and not on one op.

## 7. Reshape `op_case` heuristic misreads shapes

At `ggml-decoder.cpp:288`, reshape of [1,1,256,256] to [256,256] matches `op_case = 3`, and the translator builds the shape {1,1,-1,1}. That gives a MatMul shape error.

`master` skips this heuristic for the naive path (`if (m_naive) break;`), so no new report is necessary.

## Not defects

- **Joint logits of about +/-3000.** These are normal raw logits for this model.
- **F16 precision.** F16 is not the cause of the transcript errors. At the failing node, F16 on the OpenVINO CPU device and the NPU with the bias form both have an error of about 0.1%.

## Limits of this evidence

- There is no standalone reproduction. All results come from NeMo-Speech.cpp with a throwaway build, on one 11 s clip.
- I did not run `test-backend-ops`. A llama.cpp report will need a small reproduction for items 3 and 4, and that should be possible with a small graph.
- All the spike fixes are for diagnosis, and they are not ready to submit. For example, forcing `m_is_static` affects all small graphs.
- The build also needs `-DOV_GPU_USE_OPENCL_HPP` on this machine. This comes from OpenVINO's header choice on this system, and I did not check it on other systems.

## How to send this to llama.cpp

`AGENTS.md` in llama.cpp forbids AI-written PR descriptions, commit messages, comments and reviewer replies, and agent submission of PRs and issues. It gives closure or a contributor ban as the penalty. So you must write and post the report yourself, and you must understand each change. The project also asks contributors to search existing issues first and to discuss features before they send a PR.

Use this file as your notes. Do not paste it into llama.cpp.

## Files

- Backend diff: `spike-npu/ggml-openvino.diff`
- Engine changes: the commit that holds this file
