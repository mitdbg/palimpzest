import { SvelteComponent } from "svelte";
import type { I18nFormatter } from "@gradio/utils";
import type { WaveformOptions } from "../shared/types";
declare const __propDef: {
    props: {
        recording?: boolean | undefined;
        paused_recording?: boolean | undefined;
        stop: () => void;
        record: () => void;
        i18n: I18nFormatter;
        waveform_settings: Record<string, any>;
        waveform_options?: WaveformOptions | undefined;
        waiting?: boolean | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type StreamAudioProps = typeof __propDef.props;
export type StreamAudioEvents = typeof __propDef.events;
export type StreamAudioSlots = typeof __propDef.slots;
export default class StreamAudio extends SvelteComponent<StreamAudioProps, StreamAudioEvents, StreamAudioSlots> {
}
export {};
