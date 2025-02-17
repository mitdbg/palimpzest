import { SvelteComponent } from "svelte";
export { default as BaseMultimodalTextbox } from "./shared/MultimodalTextbox.svelte";
export { default as BaseExample } from "./Example.svelte";
import type { Gradio, SelectData } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
import type { FileData } from "@gradio/client";
declare const __propDef: {
    props: {
        gradio: Gradio<{
            change: {
                text: string;
                files: FileData[];
            };
            submit: never;
            stop: never;
            blur: never;
            select: SelectData;
            input: never;
            focus: never;
            error: string;
            clear_status: LoadingStatus;
            start_recording: never;
            pause_recording: never;
            stop_recording: never;
            upload: FileData[] | FileData;
            clear: undefined;
        }>;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: {
            text: string;
            files: FileData[];
        } | undefined;
        file_types?: (string[] | null) | undefined;
        lines: number;
        placeholder?: string | undefined;
        label?: string | undefined;
        info?: string | undefined;
        show_label: boolean;
        max_lines: number;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        submit_btn?: (string | boolean | null) | undefined;
        stop_btn?: (string | boolean | null) | undefined;
        loading_status?: LoadingStatus | undefined;
        value_is_output?: boolean | undefined;
        rtl?: boolean | undefined;
        text_align?: "left" | "right" | undefined;
        autofocus?: boolean | undefined;
        autoscroll?: boolean | undefined;
        interactive: boolean;
        root: string;
        file_count: "single" | "multiple" | "directory";
        max_plain_text_length: number;
        sources?: ["microphone" | "upload"] | undefined;
        waveform_options?: any;
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
    get gradio(): Gradio<{
        change: {
            text: string;
            files: FileData[];
        };
        submit: never;
        stop: never;
        blur: never;
        select: SelectData;
        input: never;
        focus: never;
        error: string;
        clear_status: LoadingStatus;
        start_recording: never;
        pause_recording: never;
        stop_recording: never;
        upload: FileData[] | FileData;
        clear: undefined;
    }>;
    /**accessor*/
    set gradio(_: Gradio<{
        change: {
            text: string;
            files: FileData[];
        };
        submit: never;
        stop: never;
        blur: never;
        select: SelectData;
        input: never;
        focus: never;
        error: string;
        clear_status: LoadingStatus;
        start_recording: never;
        pause_recording: never;
        stop_recording: never;
        upload: FileData[] | FileData;
        clear: undefined;
    }>);
    get elem_id(): string | undefined;
    /**accessor*/
    set elem_id(_: string | undefined);
    get elem_classes(): string[] | undefined;
    /**accessor*/
    set elem_classes(_: string[] | undefined);
    get visible(): boolean | undefined;
    /**accessor*/
    set visible(_: boolean | undefined);
    get value(): {
        text: string;
        files: FileData[];
    } | undefined;
    /**accessor*/
    set value(_: {
        text: string;
        files: FileData[];
    } | undefined);
    get file_types(): string[] | null | undefined;
    /**accessor*/
    set file_types(_: string[] | null | undefined);
    get lines(): number;
    /**accessor*/
    set lines(_: number);
    get placeholder(): string | undefined;
    /**accessor*/
    set placeholder(_: string | undefined);
    get label(): string | undefined;
    /**accessor*/
    set label(_: string | undefined);
    get info(): string | undefined;
    /**accessor*/
    set info(_: string | undefined);
    get show_label(): boolean;
    /**accessor*/
    set show_label(_: boolean);
    get max_lines(): number;
    /**accessor*/
    set max_lines(_: number);
    get scale(): number | null | undefined;
    /**accessor*/
    set scale(_: number | null | undefined);
    get min_width(): number | undefined;
    /**accessor*/
    set min_width(_: number | undefined);
    get submit_btn(): string | boolean | null | undefined;
    /**accessor*/
    set submit_btn(_: string | boolean | null | undefined);
    get stop_btn(): string | boolean | null | undefined;
    /**accessor*/
    set stop_btn(_: string | boolean | null | undefined);
    get loading_status(): LoadingStatus | undefined;
    /**accessor*/
    set loading_status(_: LoadingStatus | undefined);
    get value_is_output(): boolean | undefined;
    /**accessor*/
    set value_is_output(_: boolean | undefined);
    get rtl(): boolean | undefined;
    /**accessor*/
    set rtl(_: boolean | undefined);
    get text_align(): "left" | "right" | undefined;
    /**accessor*/
    set text_align(_: "left" | "right" | undefined);
    get autofocus(): boolean | undefined;
    /**accessor*/
    set autofocus(_: boolean | undefined);
    get autoscroll(): boolean | undefined;
    /**accessor*/
    set autoscroll(_: boolean | undefined);
    get interactive(): boolean;
    /**accessor*/
    set interactive(_: boolean);
    get root(): string;
    /**accessor*/
    set root(_: string);
    get file_count(): "single" | "multiple" | "directory";
    /**accessor*/
    set file_count(_: "single" | "multiple" | "directory");
    get max_plain_text_length(): number;
    /**accessor*/
    set max_plain_text_length(_: number);
    get sources(): ["upload" | "microphone"] | undefined;
    /**accessor*/
    set sources(_: ["upload" | "microphone"] | undefined);
    get waveform_options(): any;
    /**accessor*/
    set waveform_options(_: any);
}
