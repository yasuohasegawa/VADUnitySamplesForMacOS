using UnityEngine;

public class WebRTCTest : MonoBehaviour
{
    const int SampleRate = 16000;   // Common supported rate by WebRTC VAD
    const int FrameDurationMs = 10; // 10, 20, or 30 ms frames are supported

    [SerializeField] private MeshRenderer _recordCheckMesh;

    private WebRTCVADWrapper _webRTCVADWrapper;

    private int _frameSize;
    private AudioClip _micClip;
    private int _micPosition = 0;
    private float[] _audioBuffer;
    private short[] _pcmBuffer;

    private bool _isSpeaking = false;
    private float _speechTimeout = 0.1f;
    private float _speechTimer = 0f;

    void Start()
    {
        _webRTCVADWrapper = new WebRTCVADWrapper();
        _frameSize = SampleRate * FrameDurationMs / 1000;

        _micClip = Microphone.Start(null, true, 1, SampleRate);
        if (_micClip.frequency != 16000)
            Debug.LogWarning("Unexpected sample rate: " + _micClip.frequency);

        _audioBuffer = new float[_micClip.samples * _micClip.channels];
        _pcmBuffer = new short[_frameSize];
    }

    // Update is called once per frame
    void Update()
    {
        //return;
        int pos = Microphone.GetPosition(null);
        //Debug.Log($"pos:{pos}");
        if (pos < 0 || pos == _micPosition) return;
        

        int samplesAvailable = (pos - _micPosition + _micClip.samples) % _micClip.samples;

        _micClip.GetData(_audioBuffer, _micPosition);

        _micPosition = pos;

        int offset = 0;
        bool detectedThisFrame = false;
        while (samplesAvailable >= _frameSize)
        {
            // Convert floats [-1..1] to shorts [-32768..32767]
            float sum = 0f;
            for (int i = 0; i < _frameSize; i++)
            {
                float f = _audioBuffer[offset + i];
                f = Mathf.Clamp(f, -1f, 1f);
                _pcmBuffer[i] = (short)(f * 32767);
                sum += f * f;
            }
            float rms = Mathf.Sqrt(sum / _frameSize);
            bool isSpeech = _webRTCVADWrapper.IsSpeech(_pcmBuffer, SampleRate);
            Debug.Log($"Speech detected? {isSpeech},{rms:F4}");
            if (isSpeech && rms > 0.01f)
            {
                detectedThisFrame = true;
            }

            offset += _frameSize;
            samplesAvailable -= _frameSize;
        }

        if (detectedThisFrame)
        {
            _isSpeaking = true;
            _speechTimer = _speechTimeout;
        }
        else
        {
            _speechTimer -= Time.deltaTime;
            if (_speechTimer <= 0f)
            {
                _isSpeaking = false;
            }
        }
        //Debug.Log($"Speaking? {_isSpeaking},{_speechTimer}");
        _recordCheckMesh.material.color = _isSpeaking ? Color.red : Color.white;
    }

    private void ForceStop()
    {
        Microphone.End(null);
        _webRTCVADWrapper = null;
    }

    private void OnDestroy()
    {
        ForceStop();
    }

    private void OnApplicationQuit()
    {
        ForceStop();
    }
}
