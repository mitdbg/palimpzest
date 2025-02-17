import type WaveSurfer from "wavesurfer.js";
export interface LoadedParams {
    autoplay?: boolean;
}
export declare function blob_to_data_url(blob: Blob): Promise<string>;
export declare const process_audio: (audioBuffer: AudioBuffer, start?: number, end?: number, waveform_sample_rate?: number) => Promise<Uint8Array>;
export declare function loaded(node: HTMLAudioElement, { autoplay }?: LoadedParams): void;
export declare const skip_audio: (waveform: WaveSurfer, amount: number) => void;
export declare const get_skip_rewind_amount: (audio_duration: number, skip_length?: number | null) => number;
