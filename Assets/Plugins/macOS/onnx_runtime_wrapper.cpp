#include <stdio.h>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <onnxruntime_c_api.h>

extern "C" const OrtApi* OrtGetApiWrapper() {
    return OrtGetApiBase()->GetApi(ORT_API_VERSION);
}

static const OrtApi* g_ort = NULL;
static OrtEnv* g_env = NULL;
static OrtSession* g_session = NULL;
static OrtSessionOptions* g_session_options = NULL;

extern "C" int LoadModel(const char* model_path) {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "silero", &g_env) != NULL) return -1;
    if (g_ort->CreateSessionOptions(&g_session_options) != NULL) return -2;
    if (g_ort->CreateSession(g_env, model_path, g_session_options, &g_session) != NULL) return -3;
    return 0;
}

extern "C" int IsSpeech(float* input_data, int length, float* output) {
    OrtMemoryInfo* memory_info = NULL;
    OrtStatus* status = NULL;
    OrtValue *input_tensor = NULL, *state_tensor = NULL, *sr_tensor = NULL;
    OrtValue *output_tensors[2] = { NULL, NULL };

    // Create MemoryInfo
    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status != NULL) {
        fprintf(stderr, "CreateCpuMemoryInfo failed: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return -1;
    }

    // Create input tensor (audio frame)
    int64_t input_shape[] = {1, length};
    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, input_data, sizeof(float) * length,
        input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    if (status != NULL) {
        fprintf(stderr, "Create input_tensor failed: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return -2;
    }

    // Create zeroed state tensor
    static float state_data[2 * 1 * 128] = {0};
    int64_t state_shape[] = {2, 1, 128};
    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, state_data, sizeof(state_data),
        state_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &state_tensor);
    if (status != NULL) {
        fprintf(stderr, "Create state_tensor failed: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return -3;
    }

    // Create sample rate tensor
    int64_t sr_value = 16000;
    int64_t sr_shape[] = {};
    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, &sr_value, sizeof(sr_value),
        sr_shape, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &sr_tensor);
    if (status != NULL) {
        fprintf(stderr, "Create sr_tensor failed: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return -4;
    }

    // Run inference
    const char* input_names[] = {"input", "state", "sr"};
    const char* output_names[] = {"output", "stateN"};

    const OrtValue* input_tensors[3] = { input_tensor, state_tensor, sr_tensor };

    status = g_ort->Run(g_session, NULL,
                        input_names, input_tensors, 3,
                        output_names, 2, output_tensors);
    if (status != NULL) {
        fprintf(stderr, "Run failed: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return -5;
    }

    // Read output value (speech score)
    float* out_data = NULL;
    status = g_ort->GetTensorMutableData(output_tensors[0], (void**)&out_data);
    if (status != NULL) {
        fprintf(stderr, "GetTensorMutableData failed: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return -6;
    }

    float* new_state = NULL;
    status = g_ort->GetTensorMutableData(output_tensors[1], (void**)&new_state);
    if (status != NULL) {
        fprintf(stderr, "GetTensorMutableData failed: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return -7;
    }
    memcpy(state_data, new_state, sizeof(state_data));

    *output = out_data[0];  // speech probability

    // Keep updated state for future frame (optional: copy from output_tensors[1] to state_data if needed)

    // Cleanup
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(state_tensor);
    g_ort->ReleaseValue(sr_tensor);
    g_ort->ReleaseValue(output_tensors[0]);
    g_ort->ReleaseValue(output_tensors[1]);
    g_ort->ReleaseMemoryInfo(memory_info);

    return 0;
}

extern "C" void Cleanup() {
    if (g_session) {
        g_ort->ReleaseSession(g_session);
        g_session = NULL;
    }
    if (g_session_options) {
        g_ort->ReleaseSessionOptions(g_session_options);
        g_session_options = NULL;
    }
    if (g_env) {
        g_ort->ReleaseEnv(g_env);
        g_env = NULL;
    }
}

extern "C" struct SpeechSegment {
    float start;
    float end;
};

extern "C" int GetSpeechTimestamps(
    float* audio,
    int length,
    int sample_rate,
    float threshold,
    float neg_threshold,
    int min_speech_ms,
    float max_speech_s,
    int min_silence_ms,
    int pad_ms,
    int return_seconds,
    int resolution,
    float** result_array,
    int* result_count)
{
    if (!audio || length <= 0 || !result_array || !result_count) return -1;

    int window_size = (sample_rate == 16000) ? 512 : 256;
    int step = 1;

    if (sample_rate > 16000 && (sample_rate % 16000 == 0)) {
        step = sample_rate / 16000;
        sample_rate = 16000;
        window_size = 512;
    }

    float min_speech_samples = sample_rate * min_speech_ms / 1000.0f;
    float max_speech_samples = sample_rate * max_speech_s - window_size - 2 * (sample_rate * pad_ms / 1000.0f);
    float min_silence_samples = sample_rate * min_silence_ms / 1000.0f;
    float min_silence_samples_at_max = sample_rate * 0.098f;
    float pad_samples = sample_rate * pad_ms / 1000.0f;
    if (neg_threshold < 0)
        neg_threshold = std::max(threshold - 0.15f, 0.01f);

    std::vector<float> speech_probs;
    for (int i = 0; i < length; i += window_size) {
        float chunk[512] = {0};
        int len = std::min(window_size, length - i);
        memcpy(chunk, audio + i, sizeof(float) * len);

        float prob = 0.0f;
        int res = IsSpeech(chunk, window_size, &prob);
        if (res != 0) return -2;
        speech_probs.push_back(prob);
    }

    std::vector<SpeechSegment> speeches;
    bool triggered = false;
    SpeechSegment current = {0, 0};
    int temp_end = 0, prev_end = 0, next_start = 0;

    for (size_t i = 0; i < speech_probs.size(); ++i) {
        float prob = speech_probs[i];
        int pos = i * window_size;

        if (prob >= threshold && temp_end > 0) {
            temp_end = 0;
            if (next_start < prev_end)
                next_start = pos;
        }

        if (prob >= threshold && !triggered) {
            triggered = true;
            current.start = (float)pos;
            continue;
        }

        if (triggered && (pos - current.start) > max_speech_samples) {
            if (prev_end > 0) {
                current.end = (float)prev_end;
                speeches.push_back(current);
                triggered = next_start >= prev_end;
                if (triggered)
                    current.start = (float)next_start;
                current = {0, 0};
                prev_end = next_start = temp_end = 0;
            } else {
                current.end = (float)pos;
                speeches.push_back(current);
                current = {0, 0};
                triggered = false;
                prev_end = next_start = temp_end = 0;
                continue;
            }
        }

        if (prob < neg_threshold && triggered) {
            if (temp_end == 0)
                temp_end = pos;
            if ((pos - temp_end) > min_silence_samples_at_max)
                prev_end = temp_end;
            if ((pos - temp_end) < min_silence_samples)
                continue;
            current.end = (float)temp_end;
            if ((current.end - current.start) > min_speech_samples)
                speeches.push_back(current);
            current = {0, 0};
            triggered = false;
            prev_end = next_start = temp_end = 0;
        }
    }

    if (triggered && (length - current.start) > min_speech_samples) {
        current.end = (float)length;
        speeches.push_back(current);
    }

    // Padding
    for (size_t i = 0; i < speeches.size(); ++i) {
        if (i == 0)
            speeches[i].start = std::max(0.0f, speeches[i].start - pad_samples);
        if (i < speeches.size() - 1) {
            float silence = speeches[i + 1].start - speeches[i].end;
            if (silence < 2 * pad_samples) {
                float half = silence / 2;
                speeches[i].end += half;
                speeches[i + 1].start = std::max(0.0f, speeches[i + 1].start - half);
            } else {
                speeches[i].end += pad_samples;
                speeches[i + 1].start = std::max(0.0f, speeches[i + 1].start - pad_samples);
            }
        } else {
            speeches[i].end += pad_samples;
        }
    }

    // Final formatting
    *result_count = (int)speeches.size();
    *result_array = (float*)malloc(sizeof(float) * 2 * (*result_count));
    for (int i = 0; i < *result_count; ++i) {
        float start = speeches[i].start;
        float end = speeches[i].end;
        if (return_seconds) {
            start = roundf(start / sample_rate * powf(10, resolution)) / powf(10, resolution);
            end = roundf(end / sample_rate * powf(10, resolution)) / powf(10, resolution);
        } else if (step > 1) {
            start *= step;
            end *= step;
        }
        (*result_array)[2 * i] = start;
        (*result_array)[2 * i + 1] = end;
    }

    return 0;
}

extern "C" void FreeMemory(void* ptr) {
    free(ptr);
}