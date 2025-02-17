import { SvelteComponent } from "svelte";
import type { I18nFormatter } from "@gradio/utils";
import type { WaveformOptions } from "../shared/types";
declare const __propDef: {
    props: {
        mode: string;
        i18n: I18nFormatter;
        dispatch_blob: (blobs: Uint8Array[] | Blob[], event: "stream" | "change" | "stop_recording") => Promise<void> | undefined;
        waveform_settings: Record<string, any>;
        waveform_options?: WaveformOptions | undefined;
        handle_reset_value: () => void;
        editable?: boolean | undefined;
        recording?: boolean | undefined;
    };
    events: {
        start_recording: CustomEvent<undefined>;
        pause_recording: CustomEvent<undefined>;
        stop_recording: CustomEvent<undefined>;
        stop: CustomEvent<undefined>;
        play: CustomEvent<undefined>;
        pause: CustomEvent<undefined>;
        end: CustomEvent<undefined>;
        edit: CustomEvent<undefined>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type AudioRecorderProps = typeof __propDef.props;
export type AudioRecorderEvents = typeof __propDef.events;
export type AudioRecorderSlots = typeof __propDef.slots;
export default class AudioRecorder extends SvelteComponent<AudioRecorderProps, AudioRecorderEvents, AudioRecorderSlots> {
}
export {};
