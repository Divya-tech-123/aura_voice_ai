/**
 * Audio Recording Service for AURA Voice UI.
 *
 * Uses the browser's standard MediaRecorder API for reliable microphone capture
 * across modern Chromium (Chrome, Edge), Firefox, and Safari on Windows/macOS/Linux.
 *
 * Encodes audio using the best natively supported compressed format (WebM/Opus, OGG, or MP4),
 * which is then cleanly converted server-side by AURA's backend to 16 kHz mono PCM WAV
 * via FFmpeg for high-accuracy speech recognition.
 */

export class AudioRecorder {
  private mediaStream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private audioTrack: MediaStreamTrack | null = null;
  private recordedChunks: Blob[] = [];
  private isRecording = false;
  private startTime = 0;
  private selectedMimeType = '';

  /**
   * Check if microphone capture and MediaRecorder are supported by the current browser.
   */
  public static isSupported(): boolean {
    return !!(
      typeof window !== 'undefined' &&
      navigator.mediaDevices &&
      typeof navigator.mediaDevices.getUserMedia === 'function' &&
      typeof window.MediaRecorder !== 'undefined'
    );
  }

  /**
   * Request microphone permission, inspect audio hardware, and begin MediaRecorder capture.
   */
  public async start(): Promise<void> {
    if (this.isRecording) {
      console.warn('[AudioRecorder] Recording is already active.');
      return;
    }

    if (!AudioRecorder.isSupported()) {
      throw new Error(
        'MediaRecorder is not supported in this browser. Please use a modern browser such as Chrome, Edge, or Firefox.'
      );
    }

    this.recordedChunks = [];
    this.startTime = 0;

    // 1. Request microphone access
    console.log('[AudioRecorder] Requesting microphone...');
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });
    } catch (err: any) {
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        throw new Error(
          'Microphone permission was denied. Please allow microphone access in your browser settings to speak to AURA.'
        );
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        throw new Error('No microphone device was found on your system. Please connect a microphone and try again.');
      } else {
        throw new Error(`Microphone access error: ${err.message || 'Hardware unavailable'}`);
      }
    }

    console.log('[AudioRecorder] Microphone permission granted');

    // 2. Verify returned audio tracks and inspect settings
    const audioTracks = this.mediaStream.getAudioTracks();
    console.log(`[AudioRecorder] Audio tracks: ${audioTracks.length}`);

    if (!audioTracks || audioTracks.length === 0) {
      this.cleanup();
      throw new Error('No audio tracks found in the microphone media stream.');
    }

    this.audioTrack = audioTracks[0];

    // Ensure audio track is enabled
    if (!this.audioTrack.enabled) {
      console.warn('[AudioRecorder] Track was disabled, enabling track...');
      this.audioTrack.enabled = true;
    }

    // Print track diagnostics (Requirement 4)
    console.log('[AudioRecorder] track.enabled:', this.audioTrack.enabled);
    console.log('[AudioRecorder] track.muted:', this.audioTrack.muted);
    console.log('[AudioRecorder] track.readyState:', this.audioTrack.readyState);
    try {
      const settings = this.audioTrack.getSettings();
      console.log('[AudioRecorder] track.getSettings():', JSON.stringify(settings));
    } catch (settingsErr) {
      console.warn('[AudioRecorder] track.getSettings() error:', settingsErr);
    }

    // Verify track is live
    if (this.audioTrack.readyState !== 'live') {
      const state = this.audioTrack.readyState;
      this.cleanup();
      throw new Error(`Microphone audio track was not live (state='${state}'). Please check your microphone connection.`);
    }

    // Identify active microphone device label
    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter((d) => d.kind === 'audioinput');
        const activeDeviceId = this.audioTrack.getSettings()?.deviceId;
        const matchedDevice = audioInputs.find((d) => d.deviceId === activeDeviceId);
        console.log(`[AudioRecorder] Selected microphone: "${matchedDevice ? matchedDevice.label : (activeDeviceId || 'Default')}"`);
      } catch (devErr) {
        console.warn('[AudioRecorder] Device enumeration error:', devErr);
      }
    }

    // 3. Dynamically select supported MediaRecorder MIME type (Requirement 5)
    const mimeTypes = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
    ];

    const supportedType = mimeTypes.find((type) => {
      try {
        return MediaRecorder.isTypeSupported(type);
      } catch {
        return false;
      }
    });

    this.selectedMimeType = supportedType || '';
    console.log(`[AudioRecorder] MIME type: ${this.selectedMimeType || 'browser-default'}`);

    // 4. Initialize MediaRecorder (Requirement 6 & 7)
    try {
      const options: MediaRecorderOptions = {};
      if (this.selectedMimeType) {
        options.mimeType = this.selectedMimeType;
      }
      this.mediaRecorder = new MediaRecorder(this.mediaStream, options);
    } catch (createErr: any) {
      console.warn('[AudioRecorder] Error creating MediaRecorder with options, attempting default:', createErr);
      try {
        this.mediaRecorder = new MediaRecorder(this.mediaStream);
        this.selectedMimeType = this.mediaRecorder.mimeType || '';
      } catch (fallbackErr: any) {
        this.cleanup();
        throw new Error(`Failed to initialize MediaRecorder: ${fallbackErr.message || fallbackErr}`);
      }
    }

    // Collect recorded audio chunks on dataavailable
    this.mediaRecorder.ondataavailable = (event: BlobEvent) => {
      if (event.data && event.data.size > 0) {
        this.recordedChunks.push(event.data);
        console.log(`[AudioRecorder] Received audio chunk: ${event.data.size} bytes`);
      }
    };

    // Attach track event listeners
    this.audioTrack.onmute = () => {
      console.warn('[AudioRecorder] audioTrack.onmute - Microphone muted at hardware/OS level.');
    };
    this.audioTrack.onunmute = () => {
      console.log('[AudioRecorder] audioTrack.onunmute - Microphone unmuted.');
    };
    this.audioTrack.onended = () => {
      console.warn('[AudioRecorder] audioTrack.onended - Audio track ended.');
    };

    // 5. Start recording with 250ms timeslice to gather regular chunks
    this.mediaRecorder.start(250);
    this.isRecording = true;
    this.startTime = Date.now();
    console.log('[AudioRecorder] Recording started');
  }

  /**
   * Stop recording, clean up hardware tracks, and assemble the audio Blob.
   */
  public async stop(): Promise<Blob> {
    if (!this.isRecording || !this.mediaRecorder) {
      throw new Error('Recorder is not active.');
    }

    this.isRecording = false;
    const durationSec = (Date.now() - this.startTime) / 1000;
    const trackMuted = this.audioTrack?.muted ?? false;
    const resolvedMimeType = this.selectedMimeType || this.mediaRecorder.mimeType || 'audio/webm';

    console.log('[AudioRecorder] Recording stopped');

    return new Promise<Blob>((resolve, reject) => {
      const recorder = this.mediaRecorder;
      if (!recorder) {
        this.cleanup();
        reject(new Error('MediaRecorder was destroyed before stopping.'));
        return;
      }

      recorder.onstop = () => {
        try {
          // Check duration (Requirement 9: at least 0.3 seconds)
          if (durationSec < 0.3) {
            this.cleanup();
            reject(new Error('Recording was too short. Please speak clearly for at least a second.'));
            return;
          }

          // Assemble recorded Blob (Requirement 8)
          const audioBlob = new Blob(this.recordedChunks, { type: resolvedMimeType });
          console.log(`[AudioRecorder] Final blob: ${audioBlob.size} bytes, type: ${audioBlob.type}`);

          // Validate non-empty Blob (Requirement 9)
          if (!audioBlob || audioBlob.size === 0) {
            this.cleanup();
            if (trackMuted) {
              reject(
                new Error(
                  'Microphone is muted at the system or hardware level. Please unmute your microphone in Windows settings or on your headset.'
                )
              );
            } else {
              reject(
                new Error(
                  'Recorded audio was empty. Please check that your microphone is working and speak clearly.'
                )
              );
            }
            return;
          }

          this.cleanup();
          resolve(audioBlob);
        } catch (assembleErr) {
          this.cleanup();
          reject(assembleErr);
        }
      };

      try {
        if (recorder.state !== 'inactive') {
          recorder.stop();
        }
      } catch (stopErr) {
        this.cleanup();
        reject(stopErr);
      }
    });
  }

  /**
   * Cancel ongoing recording and release hardware without returning audio.
   */
  public cancel(): void {
    this.isRecording = false;
    this.recordedChunks = [];
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      try {
        this.mediaRecorder.stop();
      } catch {
        // Ignore stop errors on cancellation
      }
    }
    this.cleanup();
  }

  /**
   * Stop microphone tracks and release media streams.
   */
  private cleanup(): void {
    if (this.mediaRecorder) {
      this.mediaRecorder.ondataavailable = null;
      this.mediaRecorder.onstop = null;
      this.mediaRecorder = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }
    this.audioTrack = null;
    this.recordedChunks = [];
  }
}
