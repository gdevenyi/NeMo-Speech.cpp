I want to add an experimental NPU backend for streaming ASR. Before I do more work, I want to know if you will accept it, and how you want it structured.

## Goal

Run streaming ASR well on machines that have an NPU but no useful GPU. The first step has a small scope:

- Linux on Intel Lunar Lake only. This is the only hardware I can test.
- The Nemotron 3.5 cache-aware streaming encoder at a fixed batch size. Its chunks have static shapes, which the NPU needs.
- Mel features and the RNNT predictor and joint stay on the CPU.
- The `ggml-openvino` backend that is already in the pinned llama.cpp, so that no second execution path is necessary.
- A source build with an `openvino-asr` preset. No release archives and no changes to the install scripts.
- An explicit `--device npu`. `auto` does not select it. If the NPU cannot run the request, the engine stops with an error and does not change to the CPU silently.
- A CI job that builds the preset and runs the model-free tests. CI has no NPU, so I report NPU results from my own machine.

## Current state

I have a throwaway prototype, not a PR. The encoder runs on the NPU, with the predictor and joint on the CPU, and the transcript is correct. **It runs in real time at 1.12 s chunks, but not at 160 ms chunks.**

Test set-up: Core Ultra 7 258V, OpenVINO 2026.4.1, NPU driver 1.38.0, llama.cpp bd4f514, `nemotron-3.5-asr-streaming-0.6b.q8_0.gguf`, one stream, `--asr.batching.state_arena_slots 1`, the 11 s `jfk.wav` clip. The table below uses that one short clip.

| Configuration | Chunk | Chunk avg (ms) | Chunk p99 (ms) | RTFx | Transcript |
|---|---|---|---|---|---|
| ggml CPU | 160 ms | 33 | 44 | 4.69 | Correct |
| OpenVINO, CPU device, F32 | 160 ms | 101 | 147 | 1.53 | Correct |
| OpenVINO, NPU encoder | 160 ms | 466 | 475 | 0.33 | Correct |
| ggml CPU | 1.12 s | 101 | 125 | 8.95 | Correct |
| OpenVINO, NPU encoder | 1.12 s | 854 | 971 | 1.05 | Correct |

I also ran the 100-utterance subset of LibriSpeech test-clean from `BENCHMARK.md` (every 26th utterance, 13.3 min) at 1.12 s chunks, scored with the Whisper English normalizer:

| Backend | WER | Chunk avg (ms) | Chunk p99 (ms) | RTFx |
|---|---|---|---|---|
| ggml CPU | 4.458% | 100 | 145 | 8.19 |
| OpenVINO, NPU encoder | 4.458% | 837 | 979 | 0.96 |

Only 3 of the 100 transcripts differ between the two. The p99 chunk time is below the 1120 ms chunk length, but the overall throughput is slightly below real time, so the NPU has no margin. These 100 clips are short, so this total includes much end-of-utterance work. I have not measured how much.

I have not measured energy, and I do not have a completed long-stream result. I stopped an 80-minute stream after about a quarter of the audio. It had no NPU kernel errors to that point, but that is too short to count as a stability result.

The form of the quantized weights has a large effect on the NPU. The backend has a bias form and a zero-point form. The results above use the bias form. With the zero-point form on all weights, the NPU encoder is 2.5x faster at 160 ms chunks (187 ms average, 202 ms p99), but the transcript has errors:

> and so my fellow americans ask not what what what what what what what what what what's your country can do for you ass for you can do for your country

I traced the error to the first large `MUL_MAT`, the `pre_encode.out` projection. Against the OpenVINO CPU device at F32, its output error is 8-14% with the zero-point form and about 0.1% with the bias form. F16 on the CPU device also gives about 0.1%, so F16 precision is not the cause. I do not know yet why the zero-point form loses accuracy on the NPU.

The CPU backend is much faster on this machine. So the case for an NPU depends on power and CPU load, and I have not measured those yet.

## What I had to change

The prototype needed these changes in this repository:

- Restore graph node sources before each scheduler allocation. `ggml_backend_sched` rewrites sources in place to its split copies, and `Session` reuses cached graphs. On later runs the scheduler sees a CPU leaf and makes no copy, so the argmax after a split reads the first run's logits. This can affect any backend whose reused graph has a split, but I have tested it only with OpenVINO.
- Keep mel on the host FFT. `src/asr/features/fe.cpp` builds the mel graph on any GPU-type backend, and `ggml-openvino` reports the NPU as a GPU.

The prototype also needed these changes in `ggml-openvino`, because its NPU path is written for LLM graphs:

- Route graphs with no `inp_pos` through the generic graph path, and cache the compiled graphs.
- Stop using the LLM reshape heuristics on these graphs. llama.cpp `master` already has this fix for the generic path.
- Slice in VIEW when strides differ only on axes of size 1.
- Treat a RESHAPE or VIEW that no node uses as a graph output.

A bump to llama.cpp `master` alone did not solve this. It stopped at the same VIEW failure.

## Questions

1. Will you accept an experimental backend for non-NVIDIA hardware in this repository, if it is off by default and has a limited scope?
2. If yes, what results do you need before you review a PR? I plan to require a correct transcript and WER within 0.2 points of the CPU on the 100-utterance subset in `BENCHMARK.md`, real-time p99 chunk time, lower energy than the CPU, and a 1-hour stream with no NPU faults. Is 1.12 s an acceptable chunk length for a first version?
3. Should backend fixes go to `patches/`, to llama.cpp first, or to both?
4. Do you prefer `ggml-openvino`, or separate OpenVINO code in this repository, if the state arenas turn out not to fit?
5. Is the scheduler source rewrite on reused graphs a known limit? I can send that fix as a separate PR if you want it.
6. Does the project have other NPU plans that this work should fit?

## AI use

I used Claude Code to write the prototype, run the measurements, and write this issue.

---

Edits on 2026-10-30: I replaced the earlier state in this issue. The first text said that the transcript was empty with an unknown cause, and it named joint logits of about +/-3000 as a symptom. The empty transcript was a stale scheduler copy, and those logit values are normal. A later edit named F16 as the main suspect for transcript errors, and that was wrong. The cause was the zero-point weight form, which the prototype had used as a speed change. The earlier 190 ms and 202 ms p99 figures came from that form. The latest edit adds the WER result and the state of the long-stream test.
