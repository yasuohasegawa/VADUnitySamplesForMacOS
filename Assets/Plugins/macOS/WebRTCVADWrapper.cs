using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class WebRTCVADWrapper: IDisposable
{
    const string DLL_NAME = "libwebrtcvad";

    [DllImport(DLL_NAME)]
    private static extern IntPtr WebRtcVad_Create();

    [DllImport(DLL_NAME)]
    private static extern int WebRtcVad_Init(IntPtr handle, int mode);

    [DllImport(DLL_NAME)]
    private static extern int WebRtcVad_Process(IntPtr handle, int fs, IntPtr audio_frame, int frame_length);

    [DllImport(DLL_NAME)]
    private static extern void WebRtcVad_Free(IntPtr handle);

    private IntPtr vadHandle;

    public WebRTCVADWrapper()
    {
        vadHandle = WebRtcVad_Create();
        if (vadHandle == IntPtr.Zero)
        {
            Debug.LogError("Failed to create VAD handle");
            return;
        }

        var ret = WebRtcVad_Init(vadHandle, 0);
        if (ret != 0)
        {
            Debug.LogError("VAD_Init failed with code: " + ret);
        }
        else
        {
            Debug.Log($"VAD initialized successfully:{ret}");
        }
    }

    ~WebRTCVADWrapper()
    {
        Dispose();
    }

    public bool IsSpeech(short[] audioFrame, int sampleRate)
    {
        if (vadHandle == IntPtr.Zero) return false;

        // Pin the managed array to get a pointer
        GCHandle handle = GCHandle.Alloc(audioFrame, GCHandleType.Pinned);
        try
        {
            IntPtr ptr = handle.AddrOfPinnedObject();
            int result = WebRtcVad_Process(vadHandle, sampleRate, ptr, audioFrame.Length);
            return result == 1;
        }
        finally
        {
            handle.Free();
        }
    }

    public void Dispose()
    {
        if (vadHandle != IntPtr.Zero)
        {
            WebRtcVad_Free(vadHandle);
            vadHandle = IntPtr.Zero;
        }
    }
}
