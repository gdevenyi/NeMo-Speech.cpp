# Intel NPU spike

This branch keeps a throwaway prototype and its results. It is not a PR candidate. The code has debug hooks and spike switches.

## Files

- `gate-report.md`: measurements and conclusions.
- `llama-cpp-defects.md`: notes on llama.cpp findings. llama.cpp forbids AI-written reports, so do not paste this file there.
- `issue-93-body.md`: the text of NVIDIA/NeMo-Speech.cpp#93 at the last edit.
- `ggml-openvino.diff`: changes to the backend in the build copy of llama.cpp. The diff includes debug prints.

The engine changes are in the same commit, in `CMakeLists.txt`, `app/`, `src/asr/features/fe.cpp` and `src/runtime/ggml/session.cpp`.

## Restore the build

I did not test these steps after I deleted the original build. The system needs OpenVINO, the NPU driver and OpenCL C++ headers.

```bash
git submodule update --init llama.cpp
scripts/build_sentencepiece_static.sh
cmake --preset cpu-asr -B build-ov -DGGML_OPENVINO=ON \
  -DNEMO_SPEECH_DEPENDENCY_PREFIX="$PWD/.deps" \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DOpenCL_INCLUDE_DIR=/usr/include/gegl-0.4/opencl \
  -DCMAKE_CXX_FLAGS=-DOV_GPU_USE_OPENCL_HPP
(cd build-ov/_deps/llama.cpp && patch -p1 < ../../../spike-npu/ggml-openvino.diff)
cmake --build build-ov --target nemo_speech_cli
```

The OpenCL path above is specific to the test machine. A reconfigure that refreshes `build-ov/_deps/llama.cpp` removes the backend changes, so apply the diff again after that.

## Working configuration

`SPIKE_PERF=1` selects the accurate weight form. Without it, the build uses the fast form, which gives transcript errors.

```bash
SPIKE_PERF=1 SPIKE_SMALL_CPU=1 GGML_OPENVINO_DEVICE=NPU build-ov/bin/nemo-speech bench asr test_files/asr/wav/test/jfk.wav \
  --mode stream -c 1 -n 1 --model nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
  --device npu --asr.batching.state_arena_slots 1 --asr.streaming.rnnt_right_context 13
```
