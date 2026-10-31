# NPU backend gate report

Date: 2026-10-30. This version replaces the earlier reports.

**Result: the gate at 160 ms chunks fails on latency.** With a correct transcript, p99 is 475 ms against the 160 ms limit. **The agreed fallback works:** at 1.12 s chunks the NPU gives a correct transcript in real time, with p99 971 ms against the 1120 ms chunk. WER matches the CPU. Energy and the long stream are not finished, so nothing yet shows that the NPU is better than the CPU. The user accepted 1.12 s as the gate chunk length.

## Set-up

- Intel Core Ultra 7 258V (Lunar Lake), NPU 4, 30 GiB memory.
- OpenVINO 2026.4.1, intel-npu-driver 1.38.0, llama.cpp pin bd4f514.
- Model: `nemotron-3.5-asr-streaming-0.6b.q8_0.gguf`, concurrency 1, `--asr.batching.state_arena_slots 1`.
- Audio: `test_files/asr/wav/test/jfk.wav` (11 s), one item. All results are first indications from this one clip.
- Encoder on the NPU. Predictor and joint on the OpenVINO CPU device, as Q5 decided. Mel on the host FFT.
- Branch `spike/npu-gate`. Changes are not committed.

## Gate result

| Criterion | Limit | Result at 160 ms | Result at 1.12 s (fallback) | Status |
|---|---|---|---|---|
| Encoder on the NPU | No encoder op on the CPU | Yes | Yes | Pass |
| p99 chunk time | Under the chunk length | 475 ms, limit 160 ms | 979 ms on the subset, limit 1120 ms | **Fail at 160 ms**; pass at 1.12 s, with throughput of 0.96x |
| Accuracy | WER at most 0.2 points above CPU | One clip correct; WER not measured | 4.458% on NPU and on CPU; 3 of 100 transcripts differ | Pass at 1.12 s |
| Energy per audio second | At least 30% below CPU | Not measured | Not measured | Unknown |
| 1-hour stream | No `ivpu` faults | Not run | Stopped at user request after 1104 of about 4300 encoder calls, with no `ivpu` errors to that point | Not finished |

No `ivpu` errors appeared in any run. Q7 first set the gate at 160 ms, and the user later accepted 1.12 s.

WER test: every 26th utterance of LibriSpeech test-clean, first 100 (13.32 min), Whisper English normalizer. CPU: chunk avg 100 ms, p99 145 ms, 8.19x. NPU: chunk avg 837 ms, p99 979 ms, 0.96x. These short clips add end-of-utterance work to the total. I did not measure how much.

## Results

All 160 ms runs use the NPU encoder. "Bias" and "zero-point" are the two quantized weight forms in `create_weight_node`.

| Weights in bias form | Chunk avg (ms) | Chunk p99 (ms) | RTFx | Transcript |
|---|---|---|---|---|
| None (all zero-point) | 187 | 202 | 0.83 | Many repeats of "what"; "ass" |
| `pre_encode.out` | 181 | 188 | 0.86 | Repeats; "ass" |
| `pre_encode.out`, `self_attn` | 232 | 242 | 0.67 | Repeats; "ass" |
| `pre_encode.out`, `pointwise_conv` | 222 | 231 | 0.70 | Fewer repeats; "ass" |
| `pre_encode.out`, `feed_forward` | 393 | 403 | 0.39 | One repeat of "what" |
| All (original backend setting) | 466 | 475 | 0.33 | **Correct** |

| Reference and fallback | Chunk | Chunk avg (ms) | Chunk p99 (ms) | RTFx | Transcript |
|---|---|---|---|---|---|
| ggml CPU | 160 ms | 33 | 44 | 4.69 | Correct |
| OpenVINO CPU device, F32 | 160 ms | 101 | 147 | 1.53 | Correct |
| **NPU, all bias** | **1.12 s** | **854** | **971** | **1.05** | **Correct** |
| ggml CPU | 1.12 s | 101 | 125 | 8.95 | Correct |

Warmup on the NPU in the correct configuration is 26 s at 1.12 s chunks and 56 s at 160 ms. The second figure is above the 30 s limit in Q15, so a compiled-model cache would be necessary at 160 ms.

## Cause of the transcript errors

The cause is the zero-point weight form on the NPU. **I introduced it** by setting `use_bias = false` as a speed fix. F16 is not the cause, which disproves my earlier statement.

Evidence, for `pre_encode.out` (`node_57`), error against the OpenVINO CPU device at F32 on speech chunks:

| Configuration | Relative RMS error |
|---|---|
| OpenVINO CPU device, F16 | 0.07-0.10% |
| NPU, bias form | 0.07-0.10% |
| NPU, zero-point form | 8-14% |

- Earlier nodes match closely: the node before `node_57` differs by 0.06%.
- Layer-by-layer data shows that the error starts at this node and is not error that grows with depth.
- The error is 8-14% with a peak of 47-67, on values up to about 1170.

**Not proven:** why the zero-point form loses accuracy on the NPU. The likely reason is that the compiler uses a faster, lower-precision integer path for it. That would explain both the 2x speed and the error, but I have no direct evidence. A saturation check on `node_57` could test this. Also not explained: F16 forced on the OpenVINO CPU device for all graphs gave an empty transcript. In that run the joint also used F16, and I did not investigate further.

## Why latency cannot meet 160 ms with this form

- The encoder takes about 450 ms per 160 ms chunk with the accurate form.
- `optimization-level=3` and the `PERFORMANCE` hint did not change latency, in the one configuration where I tried them.
- At 1.12 s, each chunk takes 854 ms. The cost does not scale down with fewer chunks as much as the audio length alone suggests, so the fixed attention context is probably a main cost. I did not measure that.

Options that I have not tested: F16 weights in place of `Q8_0`, `Q4_0` weights, a shorter attention context, state kept on the device, and new OpenVINO code (Q4 option b).

## Defects

See `spike-npu/llama-cpp-defects.md` for llama.cpp items.

In this repository:

1. The scheduler copy goes stale on a reused graph. `ggml_backend_sched` rewrites sources in place, and `Session` reuses cached graphs. This caused the empty transcript. The spike restores sources before each allocation. It can affect any backend whose reused graph has a split, but I tested only OpenVINO.
2. `src/asr/features/fe.cpp:330` builds the mel graph on any GPU-type backend. The spike uses the host FFT.

## Corrections to earlier statements

- F16 was my main suspect. It is not the cause.
- My `use_bias = false` "fix" caused the transcript errors. It also gave the low latency figures that I reported as progress: 190 ms, 202 ms, and those in the first version of issue #93.
- Joint logits of about +/-3000 are normal.
- The first cause that I gave for the empty transcript, a missing output, was only part of it. The stale scheduler copy was the main cause.

## Next steps

1. Measure energy. This needs read access to `/sys/class/powercap/intel-rapl:0/energy_uj`. With the CPU 8.5x faster, energy is the only remaining case for the NPU.
2. Run the 1-hour stream.
3. Wait for maintainer feedback on issue #93.

## Spike files

- Engine changes: the commit that holds this file.
- Backend changes, with debug prints: `spike-npu/ggml-openvino.diff`.
- The builds, datasets and test scripts were deleted after the tests. See `spike-npu/README.md`.
- The current build defaults to the zero-point form. Use `SPIKE_PERF=1` for the correct form. It also sets the `PERFORMANCE` hint.
- Other switches: `SPIKE_SMALL_CPU=1`, `SPIKE_BIAS_NAME=<a,b>`, `SPIKE_PRECISION=<value>`, `SPIKE_OUT_DIR=<dir>`, `SPIKE_DUMP_DIR=<dir>` (this hook can crash at the end of a run).
- Build note: this machine has no `opencl-clhpp` or `opencl-headers` package. The spike used `-DOpenCL_INCLUDE_DIR=/usr/include/gegl-0.4/opencl -DCMAKE_CXX_FLAGS=-DOV_GPU_USE_OPENCL_HPP`.

## Command for the working configuration

```bash
SPIKE_PERF=1 SPIKE_SMALL_CPU=1 GGML_OPENVINO_DEVICE=NPU build-ov/bin/nemo-speech bench asr test_files/asr/wav/test/jfk.wav \
  --mode stream -c 1 -n 1 --model /var/tmp/nemo-models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
  --device npu --asr.batching.state_arena_slots 1 --asr.streaming.rnnt_right_context 13
```
