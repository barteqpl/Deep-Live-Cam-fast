# Face Swap Models and CoreML Performance Documentation (Apple M4 Pro)

We have introduced advanced support for 256px models and optimized the erosion-blending mask generation directly within the crop face space. Below is the technical description of the models, recommendations, and benchmark results.

---

## 1. Available Models and Recommendations

Four main face swap models are supported in the project. Model selection is done using the `--swapper-model` parameter (e.g., `--swapper-model simswap`).

| Model | Resolution | Performance Overhead | Requires Converter | Recommended Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **SimSwap 256** | 256x256 | Low | Yes | **Best for live webcam use**. Fastest inference time, preserves expression and head orientation very well. |
| **HyperSwap 256** | 256x256 | Medium | No | **Recommended as the default high-quality option**. Very strong transfer of source identity, does not require an additional converter, and produces very natural boundaries. |
| **HiFiFace 256** | 256x256 | Medium | Yes | **Best for static image/video rendering**. Generates a very soft, built-in blending mask, but strongly adheres to the target face geometry (so you might "still look like yourself"). |
| **Inswapper 128** | 128x128 | Medium | No | Default model. Low resolution results in visible blurring on modern HD cameras/videos. |

---

## 2. Performance Stats on Apple M4 Pro (CoreML)

The following statistics represent the average time of the face swap operation (frame reading, CoreML inference, blending, and mask application) for a test video file on an Apple M4 Pro:

- **SimSwap 256**: **~41.91 ms** per frame (**~23.9 FPS**).
- **Inswapper 128**: **~75.57 ms** per frame (**~13.2 FPS**).
- **HiFiFace 256**: **~85.87 ms** per frame (**~11.7 FPS**).
- **HyperSwap 256**: **~151.59 ms** per frame (**~6.6 FPS**).

> [!NOTE]
> Thanks to the implemented mask caching in [face_swapper.py](file:///Users/barteq/repos/ai/Deep-Live-Cam/modules/processors/frame/face_swapper.py#L65), the post-processing time has dropped from several milliseconds to **<0.1 ms**, which allows maintaining a frame rate limited almost exclusively by the speed of the CoreML network itself.

---

## 3. Automatic Model Downloading

Missing models are automatically downloaded on the first run of the application by the [pre_check](file:///Users/barteq/repos/ai/Deep-Live-Cam/modules/processors/frame/face_swapper.py#L334) function from Hugging Face servers:
- For **HiFiFace**: the main file `hififace_unofficial_256.onnx` and the embedding converter `crossface_hififace.onnx` are downloaded.
- For **SimSwap**: the main file `simswap_256.onnx` and the embedding converter `crossface_simswap.onnx` are downloaded.
- Files are saved directly in the `models/` project directory.
