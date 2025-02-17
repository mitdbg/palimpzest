import { SvelteComponent } from "svelte";
import type { I18nFormatter } from "@gradio/utils";
import type { FileData } from "@gradio/client";
import type { WaveformOptions } from "../shared/types";
declare const __propDef: {
    props: {
        value?: null | FileData;
        label: string;
        show_label?: boolean | undefined;
        show_download_button?: boolean | undefined;
        show_share_button?: boolean | undefined;
        i18n: I18nFormatter;
        waveform_settings?: Record<string, any> | undefined;
        waveform_options?: WaveformOptions | undefined;
        editable?: boolean | undefined;
        loop: boolean;
        display_icon_button_wrapper_top_corner?: boolean | undefined;
    };
    events: {
        error: CustomEvent<string>;
        share: CustomEvent<import("@gradio/utils").ShareData>;
        pause: CustomEvent<any>;
        play: CustomEvent<any>;
        stop: CustomEvent<any>;
        load: CustomEvent<any>;
        change: CustomEvent<FileData>;
        end: CustomEvent<undefined>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type StaticAudioProps = typeof __propDef.props;
export type StaticAudioEvents = typeof __propDef.events;
export type StaticAudioSlots = typeof __propDef.slots;
export default class StaticAudio extends SvelteComponent<StaticAudioProps, StaticAudioEvents, StaticAudioSlots> {
}
export {};
