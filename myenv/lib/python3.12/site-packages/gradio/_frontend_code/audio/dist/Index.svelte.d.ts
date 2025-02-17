import { SvelteComponent } from "svelte";
import type { Gradio, ShareData } from "@gradio/utils";
import type { FileData } from "@gradio/client";
import type { LoadingStatus } from "@gradio/statustracker";
import type { WaveformOptions } from "./shared/types";
declare const __propDef: {
    props: {
        value_is_output?: boolean | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        interactive: boolean;
        value?: null | FileData;
        sources: ["microphone"] | ["upload"] | ["microphone", "upload"] | ["upload", "microphone"];
        label: string;
        root: string;
        show_label: boolean;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        loading_status: LoadingStatus;
        autoplay?: boolean | undefined;
        loop?: boolean | undefined;
        show_download_button: boolean;
        show_share_button?: boolean | undefined;
        editable?: boolean | undefined;
        waveform_options?: WaveformOptions | undefined;
        pending: boolean;
        streaming: boolean;
        stream_every: number;
        input_ready: boolean;
        recording?: boolean | undefined;
        modify_stream_state?: ((state: "open" | "closed" | "waiting") => void) | undefined;
        get_stream_state?: (() => void) | undefined;
        set_time_limit: (time: number) => void;
        gradio: Gradio<{
            input: never;
            change: any;
            stream: any;
            error: string;
            warning: string;
            edit: never;
            play: never;
            pause: never;
            stop: never;
            end: never;
            start_recording: never;
            pause_recording: never;
            stop_recording: never;
            upload: never;
            clear: never;
            share: ShareData;
            clear_status: LoadingStatus;
            close_stream: string;
        }>;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type IndexProps = typeof __propDef.props;
export type IndexEvents = typeof __propDef.events;
export type IndexSlots = typeof __propDef.slots;
export default class Index extends SvelteComponent<IndexProps, IndexEvents, IndexSlots> {
    get modify_stream_state(): (state: "open" | "closed" | "waiting") => void;
    get get_stream_state(): () => void;
    get value_is_output(): boolean | undefined;
    /**accessor*/
    set value_is_output(_: boolean | undefined);
    get elem_id(): string | undefined;
    /**accessor*/
    set elem_id(_: string | undefined);
    get elem_classes(): string[] | undefined;
    /**accessor*/
    set elem_classes(_: string[] | undefined);
    get visible(): boolean | undefined;
    /**accessor*/
    set visible(_: boolean | undefined);
    get interactive(): boolean;
    /**accessor*/
    set interactive(_: boolean);
    get value(): any;
    /**accessor*/
    set value(_: any);
    get sources(): ["microphone"] | ["upload"] | ["microphone", "upload"] | ["upload", "microphone"];
    /**accessor*/
    set sources(_: ["microphone"] | ["upload"] | ["microphone", "upload"] | ["upload", "microphone"]);
    get label(): string;
    /**accessor*/
    set label(_: string);
    get root(): string;
    /**accessor*/
    set root(_: string);
    get show_label(): boolean;
    /**accessor*/
    set show_label(_: boolean);
    get container(): boolean | undefined;
    /**accessor*/
    set container(_: boolean | undefined);
    get scale(): number | null | undefined;
    /**accessor*/
    set scale(_: number | null | undefined);
    get min_width(): number | undefined;
    /**accessor*/
    set min_width(_: number | undefined);
    get loading_status(): LoadingStatus;
    /**accessor*/
    set loading_status(_: LoadingStatus);
    get autoplay(): boolean | undefined;
    /**accessor*/
    set autoplay(_: boolean | undefined);
    get loop(): boolean | undefined;
    /**accessor*/
    set loop(_: boolean | undefined);
    get show_download_button(): boolean;
    /**accessor*/
    set show_download_button(_: boolean);
    get show_share_button(): boolean | undefined;
    /**accessor*/
    set show_share_button(_: boolean | undefined);
    get editable(): boolean | undefined;
    /**accessor*/
    set editable(_: boolean | undefined);
    get waveform_options(): WaveformOptions | undefined;
    /**accessor*/
    set waveform_options(_: WaveformOptions | undefined);
    get pending(): boolean;
    /**accessor*/
    set pending(_: boolean);
    get streaming(): boolean;
    /**accessor*/
    set streaming(_: boolean);
    get stream_every(): number;
    /**accessor*/
    set stream_every(_: number);
    get input_ready(): boolean;
    /**accessor*/
    set input_ready(_: boolean);
    get recording(): boolean | undefined;
    /**accessor*/
    set recording(_: boolean | undefined);
    get undefined(): any;
    /**accessor*/
    set undefined(_: any);
    get set_time_limit(): (time: number) => void;
    /**accessor*/
    set set_time_limit(_: (time: number) => void);
    get gradio(): Gradio<{
        input: never;
        change: any;
        stream: any;
        error: string;
        warning: string;
        edit: never;
        play: never;
        pause: never;
        stop: never;
        end: never;
        start_recording: never;
        pause_recording: never;
        stop_recording: never;
        upload: never;
        clear: never;
        share: ShareData;
        clear_status: LoadingStatus;
        close_stream: string;
    }>;
    /**accessor*/
    set gradio(_: Gradio<{
        input: never;
        change: any;
        stream: any;
        error: string;
        warning: string;
        edit: never;
        play: never;
        pause: never;
        stop: never;
        end: never;
        start_recording: never;
        pause_recording: never;
        stop_recording: never;
        upload: never;
        clear: never;
        share: ShareData;
        clear_status: LoadingStatus;
        close_stream: string;
    }>);
}
export {};
