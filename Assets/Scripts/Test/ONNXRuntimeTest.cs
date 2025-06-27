using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;

public class ONNXRuntimeTest : MonoBehaviour
{
    private const string DLL_NAME = "onnx_runtime_wrapper";

    [DllImport(DLL_NAME)]
    private static extern IntPtr OrtGetApiWrapper();

    [DllImport(DLL_NAME)]
    private static extern int LoadModel(string modelPath);

    [DllImport(DLL_NAME)]
    private static extern int IsSpeech(float[] input, int length, out float result);

    [DllImport(DLL_NAME)]
    private static extern void Cleanup();

    [DllImport(DLL_NAME)]
    private static extern int GetSpeechTimestamps(
    float[] audio, int length, int sampleRate,
    float threshold, float negThreshold,
    int minSpeechMs, float maxSpeechS,
    int minSilenceMs, int padMs,
    int returnSeconds, int resolution,
    out IntPtr resultArray, out int resultCount);

    [DllImport(DLL_NAME)]
    public static extern void FreeMemory(IntPtr ptr);

    private const int SampleRate = 16000;
    private const int FrameDurationMs = 30;


    [SerializeField] private MeshRenderer _recordCheckMesh;

    private AudioClip _micClip;
    private int _micPosition;
    private int _frameSize;

    private float _speechTimeout = 0.001f;
    private float _speechTimer = 0f;
    private List<float> _speechBuffer = new List<float>();
    private bool _isSpeaking = false;
    private bool _wasSpeaking = false;

    void Start()
    {
        //TestDLL();
        GetMicrophone();
        string modelPath = Application.dataPath + "/Plugins/macOS/silero_vad.onnx";
        int result = LoadModel(modelPath);
        Debug.Log("Model Load Result: " + result);

        _frameSize = SampleRate * FrameDurationMs / 1000;
        _micClip = Microphone.Start(null, true, 1, SampleRate);
        
        Debug.Log("_frameSize: " + _frameSize);

        //TestIsPeech();
    }

    void Update()
    {
        if (_micClip == null || !Microphone.IsRecording(null))
            return;

        int pos = Microphone.GetPosition(null);
        if (pos < 0 || pos == _micPosition) return;

        int samplesAvailable = (pos - _micPosition + _micClip.samples) % _micClip.samples;
        if (samplesAvailable < SampleRate / 2) return;

        float[] buffer = new float[samplesAvailable];
        _micClip.GetData(buffer, _micPosition);
        _micPosition = pos;

        // Run VAD on the full buffer
        IntPtr resultPtr;
        int count;
        int rc = GetSpeechTimestamps(
            buffer,
            buffer.Length,
            SampleRate,
            0.1f,    // threshold
            -1.0f,   // neg_threshold < 0 = auto = threshold - 0.15
            150,     // min_speech_ms
            10.0f,   // max_speech_s
            1000,     // min_silence_ms
            30,      // pad_ms
            0,       // return_seconds = false (samples)
            1,       // resolution
            out resultPtr,
            out count
        );

        bool detected = false;

        float rms = 0;
        for (int i = 0; i < buffer.Length; i++)
            rms += buffer[i] * buffer[i];
        rms = Mathf.Sqrt(rms / buffer.Length);

        if (rc == 0 && count >= 1 && rms >= 0.01f)
        {
            detected = true;
            Debug.Log($">>>>>> rc:{rc}, count:{count}, length:{buffer.Length}");
        }

        if (resultPtr != IntPtr.Zero)
        {
            FreeMemory(resultPtr);
        }

        if (detected)
        {
            _speechTimer = _speechTimeout;
            _isSpeaking = true;

            _speechBuffer.AddRange(buffer);
        }
        else
        {
            _speechTimer -= Time.deltaTime;
            Debug.Log($">>>> silence detected.{_speechTimer}");
            if (_speechTimer <= 0f && _wasSpeaking)
            {
                _isSpeaking = false;

                if (_speechBuffer.Count > SampleRate / 2)
                {
                    string path = Path.Combine(Application.persistentDataPath, $"speech_onnx.wav");
                    SaveWav(path, _speechBuffer.ToArray(), SampleRate);
                    Debug.Log($"💾 Saved speech WAV: {path}");
                }

                _speechBuffer.Clear();
            }
        }

        // Update previous speaking state
        _wasSpeaking = _isSpeaking;

        _recordCheckMesh.material.color = _isSpeaking ? Color.red : Color.white;
    }

    private string GetMicrophone()
    {
        string mic = null;
        if (Microphone.devices.Length > 0)
        {
            mic = Microphone.devices[0];
            foreach (var divice in Microphone.devices)
            {
                if (divice.Contains("Mac"))
                {
                    mic = divice;
                }
                Debug.Log(divice);
            }

            return mic;
        }
        else
        {
            Debug.LogError("No microphone detected.");
            return null;
        }
    }

    public static void SaveWav(string filePath, float[] samples, int sampleRate, int channels = 1)
    {
        using (var fileStream = new FileStream(filePath, FileMode.Create))
        {
            int sampleCount = samples.Length;
            int byteCount = sampleCount * sizeof(short); // 16-bit PCM

            // Write WAV header
            BinaryWriter writer = new BinaryWriter(fileStream);

            writer.Write(System.Text.Encoding.UTF8.GetBytes("RIFF"));
            writer.Write(36 + byteCount);
            writer.Write(System.Text.Encoding.UTF8.GetBytes("WAVE"));
            writer.Write(System.Text.Encoding.UTF8.GetBytes("fmt "));
            writer.Write(16); // Subchunk1Size (PCM)
            writer.Write((short)1); // AudioFormat (1 = PCM)
            writer.Write((short)channels);
            writer.Write(sampleRate);
            writer.Write(sampleRate * channels * sizeof(short)); // ByteRate
            writer.Write((short)(channels * sizeof(short))); // BlockAlign
            writer.Write((short)16); // BitsPerSample

            writer.Write(System.Text.Encoding.UTF8.GetBytes("data"));
            writer.Write(byteCount);

            // Write PCM samples
            foreach (float sample in samples)
            {
                short s = (short)Mathf.Clamp(sample * 32767f, -32768f, 32767f);
                writer.Write(s);
            }
        }
    }

    [ContextMenu(nameof(Dispose))]
    private void Dispose()
    {
        if (Microphone.IsRecording(null))
            Microphone.End(null);
        Cleanup();
    }

    void OnDestroy()
    {
        Dispose();
    }

    private void OnApplicationQuit()
    {
        Dispose();
    }


    /* ====== test codes ======= */

    private void TestDLL()
    {
        try
        {
            IntPtr ptr = OrtGetApiWrapper();
            Debug.Log($"OrtGetApiWrapper returned: {ptr}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to load ONNXRuntime: {ex}");
        }
    }

    private void TestIsPeech()
    {
        float[] fakeInput = new float[480];
        float sampleRate = 16000f;

        for (int i = 0; i < fakeInput.Length; i++)
        {
            float t = i / sampleRate;
            fakeInput[i] = 0.5f * Mathf.Sin(2 * Mathf.PI * 440f * t) +
                           0.25f * Mathf.Sin(2 * Mathf.PI * 880f * t) +
                           0.25f * UnityEngine.Random.Range(-0.1f, 0.1f); // subtle noise
        }
        float output;
        IsSpeech(fakeInput, fakeInput.Length, out output);
        Debug.Log("Is Speech Score: " + output);
    }
}
