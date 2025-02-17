import { SvelteComponent } from "svelte";
import type { Gradio } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        gradio: Gradio<{
            change: never;
            submit: never;
            input: never;
            clear_status: LoadingStatus;
        }>;
        label?: string | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: string | undefined;
        placeholder?: string | undefined;
        show_label: boolean;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        loading_status?: LoadingStatus | undefined;
        value_is_output?: boolean | undefined;
        interactive: boolean;
        rtl?: boolean | undefined;
        root: string;
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
        change: never;
        submit: never;
        input: never;
        clear_status: LoadingStatus;
    }>;
    /**accessor*/
    set gradio(_: Gradio<{
        change: never;
        submit: never;
        input: never;
        clear_status: LoadingStatus;
    }>);
    get label(): string | undefined;
    /**accessor*/
    set label(_: string | undefined);
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
    get placeholder(): string | undefined;
    /**accessor*/
    set placeholder(_: string | undefined);
    get show_label(): boolean;
    /**accessor*/
    set show_label(_: boolean);
    get scale(): number | null | undefined;
    /**accessor*/
    set scale(_: number | null | undefined);
    get min_width(): number | undefined;
    /**accessor*/
    set min_width(_: number | undefined);
    get loading_status(): LoadingStatus | undefined;
    /**accessor*/
    set loading_status(_: LoadingStatus | undefined);
    get value_is_output(): boolean | undefined;
    /**accessor*/
    set value_is_output(_: boolean | undefined);
    get interactive(): boolean;
    /**accessor*/
    set interactive(_: boolean);
    get rtl(): boolean | undefined;
    /**accessor*/
    set rtl(_: boolean | undefined);
    get root(): string;
    /**accessor*/
    set root(_: string);
}
export {};
