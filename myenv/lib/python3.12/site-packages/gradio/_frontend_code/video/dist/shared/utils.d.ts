import { FFmpeg } from "@ffmpeg/ffmpeg";
export declare const prettyBytes: (bytes: number) => string;
export declare const playable: () => boolean;
export declare function loaded(node: HTMLVideoElement, { autoplay }: {
    autoplay: boolean;
}): any;
export default function loadFfmpeg(): Promise<FFmpeg>;
export declare function blob_to_data_url(blob: Blob): Promise<string>;
export declare function trimVideo(ffmpeg: FFmpeg, startTime: number, endTime: number, videoElement: HTMLVideoElement): Promise<any>;
