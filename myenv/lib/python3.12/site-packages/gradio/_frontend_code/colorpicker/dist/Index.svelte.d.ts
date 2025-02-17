import { SvelteComponent } from "svelte";
export { default as BaseColorPicker } from "./shared/Colorpicker.svelte";
export { default as BaseExample } from "./Example.svelte";
import type { Gradio } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        label?: string | undefined;
        info?: string | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value: string;
        value_is_output?: boolean | undefined;
        show_label: boolean;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        loading_status: LoadingStatus;
        root: string;
        gradio: Gradio<{
            change: never;
            input: never;
            submit: never;
            blur: never;
            focus: never;
            clear_status: LoadingStatus;
        }>;
        interactive: boolean;
        disabled?: boolean | undefined;
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
    get value(): string;
    /**accessor*/
    set value(_: string);
    get value_is_output(): boolean | undefined;
    /**accessor*/
    set value_is_output(_: boolean | undefined);
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
    get root(): string;
    /**accessor*/
    set root(_: string);
    get gradio(): Gradio<{
        change: never;
        input: never;
        submit: never;
        blur: never;
        focus: never;
        clear_status: LoadingStatus;
    }>;
    /**accessor*/
    set gradio(_: Gradio<{
        change: never;
        input: never;
        submit: never;
        blur: never;
        focus: never;
        clear_status: LoadingStatus;
    }>);
    get interactive(): boolean;
    /**accessor*/
    set interactive(_: boolean);
    get disabled(): boolean | undefined;
    /**accessor*/
    set disabled(_: boolean | undefined);
}
