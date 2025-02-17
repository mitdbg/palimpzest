export declare function get_devices(): Promise<MediaDeviceInfo[]>;
export declare function handle_error(error: string): void;
export declare function set_local_stream(local_stream: MediaStream | null, video_source: HTMLVideoElement): void;
export declare function get_video_stream(include_audio: boolean, video_source: HTMLVideoElement, webcam_constraints: {
    [key: string]: any;
} | null, device_id?: string): Promise<MediaStream>;
export declare function set_available_devices(devices: MediaDeviceInfo[]): MediaDeviceInfo[];
