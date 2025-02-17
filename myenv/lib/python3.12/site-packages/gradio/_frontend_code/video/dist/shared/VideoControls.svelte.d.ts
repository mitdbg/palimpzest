import { SvelteComponent } from "svelte";
import type { FileData } from "@gradio/client";
declare const __propDef: {
    props: {
        videoElement: HTMLVideoElement;
        showRedo?: boolean | undefined;
        interactive?: boolean | undefined;
        mode?: string | undefined;
        handle_reset_value: () => void;
        handle_trim_video: (videoBlob: Blob) => void;
        processingVideo?: boolean | undefined;
        i18n: (key: string) => string;
        value?: FileData | null;
        show_download_button?: boolean | undefined;
        handle_clear?: (() => void) | undefined;
        has_change_history?: boolean | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type VideoControlsProps = typeof __propDef.props;
export type VideoControlsEvents = typeof __propDef.events;
export type VideoControlsSlots = typeof __propDef.slots;
export default class VideoControls extends SvelteComponent<VideoControlsProps, VideoControlsEvents, VideoControlsSlots> {
}
export {};
