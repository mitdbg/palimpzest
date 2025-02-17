import { SvelteComponent } from "svelte";
import { type FileData, type Client } from "@gradio/client";
import type { I18nFormatter } from "js/core/src/gradio_helper";
import type { WaveformOptions } from "../shared/types";
declare const __propDef: {
    props: {
        value?: null | FileData;
        label: string;
        root: string;
        loop: boolean;
        show_label?: boolean | undefined;
        show_download_button?: boolean | undefined;
        sources?: (["microphone"] | ["upload"] | ["microphone", "upload"] | ["upload", "microphone"]) | undefined;
        pending?: boolean | undefined;
        streaming?: boolean | undefined;
        i18n: I18nFormatter;
        waveform_settings: Record<string, any>;
        trim_region_settings?: {} | undefined;
        waveform_options?: WaveformOptions | undefined;
        dragging: boolean;
        active_source: "microphone" | "upload";
        handle_reset_value?: (() => void) | undefined;
        editable?: boolean | undefined;
        max_file_size?: (number | null) | undefined;
        upload: Client["upload"];
        stream_handler: Client["stream"];
        stream_every: number;
        uploading?: boolean | undefined;
        recording?: boolean | undefined;
        class_name?: string | undefined;
        modify_stream?: ((state: "open" | "closed" | "waiting") => void) | undefined;
        set_time_limit?: ((time: number) => void) | undefined;
    };
    events: {
        start_recording: CustomEvent<any>;
        pause_recording: CustomEvent<any>;
        stop_recording: CustomEvent<any>;
        stop: CustomEvent<any>;
        play: CustomEvent<any>;
        pause: CustomEvent<any>;
        edit: CustomEvent<any>;
        change: CustomEvent<any>;
        stream: CustomEvent<FileData>;
        end: CustomEvent<never>;
        drag: CustomEvent<boolean>;
        error: CustomEvent<string>;
        upload: CustomEvent<FileData>;
        clear: CustomEvent<undefined>;
        close_stream: CustomEvent<undefined>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type InteractiveAudioProps = typeof __propDef.props;
export type InteractiveAudioEvents = typeof __propDef.events;
export type InteractiveAudioSlots = typeof __propDef.slots;
export default class InteractiveAudio extends SvelteComponent<InteractiveAudioProps, InteractiveAudioEvents, InteractiveAudioSlots> {
    get modify_stream(): (state: "open" | "closed" | "waiting") => void;
    get set_time_limit(): (time: number) => void;
}
export {};
