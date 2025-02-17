import { SvelteComponent } from "svelte";
export { default as BaseDataFrame } from "./shared/Table.svelte";
export { default as BaseExample } from "./Example.svelte";
import type { Gradio, SelectData } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
import type { Headers, Datatype, DataframeValue } from "./shared/utils";
declare const __propDef: {
    props: {
        headers?: Headers | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: DataframeValue | undefined;
        value_is_output?: boolean | undefined;
        col_count: [number, "fixed" | "dynamic"];
        row_count: [number, "fixed" | "dynamic"];
        label?: (string | null) | undefined;
        show_label?: boolean | undefined;
        wrap: boolean;
        datatype: Datatype | Datatype[];
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        root: string;
        line_breaks?: boolean | undefined;
        column_widths?: string[] | undefined;
        gradio: Gradio<{
            change: never;
            select: SelectData;
            input: never;
            clear_status: LoadingStatus;
        }>;
        latex_delimiters: {
            left: string;
            right: string;
            display: boolean;
        }[];
        max_height?: number | undefined;
        loading_status: LoadingStatus;
        interactive: boolean;
        show_fullscreen_button?: boolean | undefined;
        show_copy_button?: boolean | undefined;
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
    get headers(): Headers | undefined;
    /**accessor*/
    set headers(_: Headers | undefined);
    get elem_id(): string | undefined;
    /**accessor*/
    set elem_id(_: string | undefined);
    get elem_classes(): string[] | undefined;
    /**accessor*/
    set elem_classes(_: string[] | undefined);
    get visible(): boolean | undefined;
    /**accessor*/
    set visible(_: boolean | undefined);
    get value(): DataframeValue | undefined;
    /**accessor*/
    set value(_: DataframeValue | undefined);
    get value_is_output(): boolean | undefined;
    /**accessor*/
    set value_is_output(_: boolean | undefined);
    get col_count(): [number, "fixed" | "dynamic"];
    /**accessor*/
    set col_count(_: [number, "fixed" | "dynamic"]);
    get row_count(): [number, "fixed" | "dynamic"];
    /**accessor*/
    set row_count(_: [number, "fixed" | "dynamic"]);
    get label(): string | null | undefined;
    /**accessor*/
    set label(_: string | null | undefined);
    get show_label(): boolean | undefined;
    /**accessor*/
    set show_label(_: boolean | undefined);
    get wrap(): boolean;
    /**accessor*/
    set wrap(_: boolean);
    get datatype(): Datatype | Datatype[];
    /**accessor*/
    set datatype(_: Datatype | Datatype[]);
    get scale(): number | null | undefined;
    /**accessor*/
    set scale(_: number | null | undefined);
    get min_width(): number | undefined;
    /**accessor*/
    set min_width(_: number | undefined);
    get root(): string;
    /**accessor*/
    set root(_: string);
    get line_breaks(): boolean | undefined;
    /**accessor*/
    set line_breaks(_: boolean | undefined);
    get column_widths(): string[] | undefined;
    /**accessor*/
    set column_widths(_: string[] | undefined);
    get gradio(): Gradio<{
        change: never;
        select: SelectData;
        input: never;
        clear_status: LoadingStatus;
    }>;
    /**accessor*/
    set gradio(_: Gradio<{
        change: never;
        select: SelectData;
        input: never;
        clear_status: LoadingStatus;
    }>);
    get latex_delimiters(): {
        left: string;
        right: string;
        display: boolean;
    }[];
    /**accessor*/
    set latex_delimiters(_: {
        left: string;
        right: string;
        display: boolean;
    }[]);
    get max_height(): number | undefined;
    /**accessor*/
    set max_height(_: number | undefined);
    get loading_status(): LoadingStatus;
    /**accessor*/
    set loading_status(_: LoadingStatus);
    get interactive(): boolean;
    /**accessor*/
    set interactive(_: boolean);
    get show_fullscreen_button(): boolean | undefined;
    /**accessor*/
    set show_fullscreen_button(_: boolean | undefined);
    get show_copy_button(): boolean | undefined;
    /**accessor*/
    set show_copy_button(_: boolean | undefined);
}
