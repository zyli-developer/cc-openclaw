import { startCapture, stopCapture } from './audio-capture';
import { createPlayer, enqueue, clearPlayback, closePlayer } from './audio-playback';

export type OnStateCallback = (state: string) => void;
export type OnTranscriptCallback = (role: 'user' | 'bot', text: string, interim: boolean) => void;
export type OnErrorCallback = (message: string) => void;

export class VoiceClient {
  private ws: WebSocket | null = null;
  private currentState: string = 'idle';

  public onState: OnStateCallback = () => {};
  public onTranscript: OnTranscriptCallback = () => {};
  public onError: OnErrorCallback = () => {};

  start(config?: { systemRole?: string; greeting?: string; comfortText?: string; mode?: string }): void {
    // Local-dev override: point at voice_gateway on :8089 (prod deploys use reverse proxy).
    // Env var takes priority; defaults to same-origin /ws for prod.
    const envUrl = process.env.NEXT_PUBLIC_VOICE_WS_URL;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = envUrl || `${protocol}//${window.location.host}/ws`;

    this.ws = new WebSocket(wsUrl);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.ws!.send(JSON.stringify({ type: 'start', ...config }));
    };

    this.ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        enqueue(new Uint8Array(event.data));
      } else {
        const msg = JSON.parse(event.data);
        if (msg.type === 'state') {
          this.currentState = msg.state;
          this.onState(msg.state);
          this._handleStateChange(msg.state);
        } else if (msg.type === 'transcript') {
          this.onTranscript(msg.role, msg.text, msg.interim);
        } else if (msg.type === 'clear_audio') {
          clearPlayback();
        } else if (msg.type === 'error') {
          this.onError(msg.message);
        }
      }
    };

    this.ws.onclose = () => {
      if (this.currentState !== 'idle' && this.currentState !== 'ending') {
        this.onError('Connection lost');
      }
      this._stopAudio();
    };

    this.ws.onerror = () => {
      this.onError('WebSocket error');
    };
  }

  stop(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'stop' }));
    }
  }

  private _handleStateChange(state: string): void {
    if (state === 'talking') {
      createPlayer();
      startCapture((chunk: Uint8Array) => {
        if (this.ws && this.ws.readyState === WebSocket.OPEN && this.currentState === 'talking') {
          this.ws.send(chunk.buffer);
        }
      });
    } else if (state === 'greeting') {
      createPlayer();
    } else if (state === 'idle' || state === 'ending' || state === 'error') {
      this._stopAudio();
    }
  }

  private _stopAudio(): void {
    stopCapture();
    closePlayer();
  }
}
