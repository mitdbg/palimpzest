import { SvelteComponent } from "svelte";
export { default as BaseTextbox } from "./shared/Textbox.svelte";
export { default as BaseExample } from "./Example.svelte";
import type { Gradio, SelectData, CopyData } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        gradio: Gradio<{
            change: string;
            submit: never;
            blur: never;
            select: SelectData;
            input: never;
            focus: never;
            stop: never;
            clear_status: LoadingStatus;
            copy: CopyData;
        }>;
        label?: string | undefined;
        info?: string | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: string | undefined;
        lines: number;
        placeholder?: string | undefined;
        show_label: boolean;
        max_lines: number;
        type?: ("text" | "password" | "email") | undefined;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        submit_btn?: (string | boolean | null) | undefined;
        stop_btn?: (string | boolean | null) | undefined;
        show_copy_button?: boolean | undefined;
        loading_status?: LoadingStatus | undefined;
        value_is_output?: boolean | undefined;
        rtl?: boolean | undefined;
        text_align?: "left" | "right" | undefined;
        autofocus?: boolean | undefined;
        autoscroll?: boolean | undefined;
        interactive: boolean;
        root: string;
        max_length?: number | undefined;
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
        change: string;
        submit: never;
        blur: never;
        select: SelectData;
        input: never;
        focus: never;
        stop: never;
        clear_status: LoadingStatus;
        copy: CopyData;
    }>;
    /**accessor*/
    set gradio(_: Gradio<{
        change: string;
        submit: never;
        blur: never;
        select: SelectData;
        input: never;
        focus: never;
        stop: never;
        clear_status: LoadingStatus;
        copy: CopyData;
    }>);
    get label(): string | undefined;
    /**accessor*/
    set label(_: string | undefined);
    get info(): string | undefined;
    /**accessor*/
    set info(_: string | undefined);
    get elem_id(): string | undefined;
    /**accessor*/
    set elem_id(_: string | undefined);
    get elem_classes(): string[] | undefined;
    /**accessor*/
    set elem_classes(_: string[] | undefined);
    get visible(): boolean | undefined;
    /**accessor*/
    set visible(_: boolean | undefined);
    get value(): string | undefined;
    /**accessor*/
    set value(_: string | undefined);
    get lines(): number;
    /**accessor*/
    set lines(_: number);
    get placeholder(): string | undefined;
    /**accessor*/
    set placeholder(_: string | undefined);
    get show_label(): boolean;
    /**accessor*/
    set show_label(_: boolean);
    get max_lines(): number;
    /**accessor*/
    set max_lines(_: number);
    get type(): "text" | "password" | "email" | undefined;
    /**accessor*/
    set type(_: "text" | "password" | "email" | undefined);
    get container(): boolean | undefined;
    /**accessor*/
    set container(_: boolean | undefined);
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
    get show_copy_button(): boolean | undefined;
    /**accessor*/
    set show_copy_button(_: boolean | undefined);
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
    get max_length(): number | undefined;
    /**accessor*/
    set max_length(_: number | undefined);
}
