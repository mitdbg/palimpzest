import { SvelteComponent } from "svelte";
import type { I18nFormatter } from "@gradio/utils";
import WaveSurfer from "wavesurfer.js";
import type { WaveformOptions } from "./types";
declare const __propDef: {
    props: {
        waveform: WaveSurfer | undefined;
        audio_duration: number;
        i18n: I18nFormatter;
        playing: boolean;
        show_redo?: boolean | undefined;
        interactive?: boolean | undefined;
        handle_trim_audio: (start: number, end: number) => void;
        mode?: string | undefined;
        container: HTMLDivElement;
        handle_reset_value: () => void;
        waveform_options?: WaveformOptions | undefined;
        trim_region_settings?: WaveformOptions | undefined;
        show_volume_slider?: boolean | undefined;
        editable?: boolean | undefined;
        trimDuration?: number | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type WaveformControlsProps = typeof __propDef.props;
export type WaveformControlsEvents = typeof __propDef.events;
export type WaveformControlsSlots = typeof __propDef.slots;
export default class WaveformControls extends SvelteComponent<WaveformControlsProps, WaveformControlsEvents, WaveformControlsSlots> {
}
export {};
