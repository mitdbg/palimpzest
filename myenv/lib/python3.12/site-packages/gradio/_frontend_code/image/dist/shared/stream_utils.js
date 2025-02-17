export function get_devices() {
    return navigator.mediaDevices.enumerateDevices();
}
export function handle_error(error) {
    throw new Error(error);
}
export function set_local_stream(local_stream, video_source) {
    video_source.srcObject = local_stream;
    video_source.muted = true;
    video_source.play();
}
export async function get_video_stream(include_audio, video_source, webcam_constraints, device_id) {
    const constraints = {
        video: device_id
            ? { deviceId: { exact: device_id }, ...webcam_constraints?.video }
            : webcam_constraints?.video || {
                width: { ideal: 1920 },
                height: { ideal: 1440 }
            },
        audio: include_audio && (webcam_constraints?.audio ?? true) // Defaults to true if not specified
    };
    return navigator.mediaDevices
        .getUserMedia(constraints)
        .then((local_stream) => {
        set_local_stream(local_stream, video_source);
        return local_stream;
    });
}
export function set_available_devices(devices) {
    const cameras = devices.filter((device) => device.kind === "videoinput");
    return cameras;
}
