# Sample Project Notes

These sample projects have been tested **only on macOS using the Unity Editor**.

If you're not aiming for the advanced VAD quality offered by services like ChatGPT or Google AI Studio, this project can serve as a solid **entry point** for implementing your own Voice Activity Detection (VAD) solution.

---

## 🎯 Overview of Included Samples

### `ONNXRuntimeTest.cs`

* This sample uses the **Silero VAD** model in ONNX format.
* It **does not** use Unity Sentis.
* Instead, it relies on **Microsoft’s ONNX Runtime** to load and execute the model.
* We **compiled ONNX Runtime from source** to create a **native Unity macOS plugin**.
* However, there appear to be some issues with our build:
  we are currently **unable to access certain features** directly from `libonnxruntime.1.22.0.dylib`.

> If you encounter similar problems, check for missing exported symbols or incorrect build configurations when compiling ONNX Runtime for Unity plugin usage.

---

### `WebRTCTest.cs`

* This sample is a **WebRTC VAD demo**.
* It includes **only the Voice Activity Detection** portion of the WebRTC source.
* It **does not include** full WebRTC functionality (e.g., audio/video streaming or signaling).