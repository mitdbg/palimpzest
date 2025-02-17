import { SvelteComponent } from "svelte";
import { type I18nFormatter } from "@gradio/utils";
import type { FileData } from "@gradio/client";
import type { WaveformOptions } from "../shared/types";
declare const __propDef: {
    props: {
        value?: null | FileData;
        label: string;
        i18n: I18nFormatter;
        dispatch_blob?: ((blobs: Uint8Array[] | Blob[], event: "stream" | "change" | "stop_recording") => Promise<void>) | undefined;
        interactive?: boolean | undefined;
        editable?: boolean | undefined;
        trim_region_settings?: {} | undefined;
        waveform_settings: Record<string, any>;
        waveform_options: WaveformOptions;
        mode?: string | undefined;
        loop: boolean;
        handle_reset_value?: (() => void) | undefined;
    };
    events: {
        load: CustomEvent<any>;
        stop: CustomEvent<undefined>;
        play: CustomEvent<undefined>;
        pause: CustomEvent<undefined>;
        edit: CustomEvent<undefined>;
        end: CustomEvent<undefined>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type AudioPlayerProps = typeof __propDef.props;
export type AudioPlayerEvents = typeof __propDef.events;
export type AudioPlayerSlots = typeof __propDef.slots;
export default class AudioPlayer extends SvelteComponent<AudioPlayerProps, AudioPlayerEvents, AudioPlayerSlots> {
}
export {};
