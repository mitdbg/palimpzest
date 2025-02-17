import { SvelteComponent } from "svelte";
import type { SelectData, CopyData } from "@gradio/utils";
declare const __propDef: {
    props: {
        value?: string | undefined;
        value_is_output?: boolean | undefined;
        lines?: number | undefined;
        placeholder?: string | undefined;
        label: string;
        info?: string | undefined;
        disabled?: boolean | undefined;
        show_label?: boolean | undefined;
        container?: boolean | undefined;
        max_lines: number;
        type?: ("text" | "password" | "email") | undefined;
        show_copy_button?: boolean | undefined;
        submit_btn?: (string | boolean | null) | undefined;
        stop_btn?: (string | boolean | null) | undefined;
        rtl?: boolean | undefined;
        autofocus?: boolean | undefined;
        text_align?: "left" | "right" | undefined;
        autoscroll?: boolean | undefined;
        max_length?: number | undefined;
        root: string;
    };
    events: {
        blur: CustomEvent<any>;
        focus: CustomEvent<any>;
        change: CustomEvent<string>;
        submit: CustomEvent<undefined>;
        stop: CustomEvent<undefined>;
        select: CustomEvent<SelectData>;
        input: CustomEvent<undefined>;
        copy: CustomEvent<CopyData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type TextboxProps = typeof __propDef.props;
export type TextboxEvents = typeof __propDef.events;
export type TextboxSlots = typeof __propDef.slots;
export default class Textbox extends SvelteComponent<TextboxProps, TextboxEvents, TextboxSlots> {
}
export {};
