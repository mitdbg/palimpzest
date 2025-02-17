import { SvelteComponent } from "svelte";
import type { SelectData } from "@gradio/utils";
import type { I18nFormatter } from "js/core/src/gradio_helper";
import { type Client } from "@gradio/client";
import type { Headers, DataframeValue, Datatype } from "./utils";
declare const __propDef: {
    props: {
        datatype: Datatype | Datatype[];
        label?: (string | null) | undefined;
        show_label?: boolean | undefined;
        headers?: Headers | undefined;
        values?: (string | number)[][] | undefined;
        col_count: [number, "fixed" | "dynamic"];
        row_count: [number, "fixed" | "dynamic"];
        latex_delimiters: {
            left: string;
            right: string;
            display: boolean;
        }[];
        editable?: boolean | undefined;
        wrap?: boolean | undefined;
        root: string;
        i18n: I18nFormatter;
        max_height?: number | undefined;
        line_breaks?: boolean | undefined;
        column_widths?: string[] | undefined;
        show_row_numbers?: boolean | undefined;
        upload: Client["upload"];
        stream_handler: Client["stream"];
        show_fullscreen_button?: boolean | undefined;
        show_copy_button?: boolean | undefined;
        value_is_output?: boolean | undefined;
        display_value?: (string[][] | null) | undefined;
        styling?: (string[][] | null) | undefined;
    };
    events: {
        change: CustomEvent<DataframeValue>;
        input: CustomEvent<undefined>;
        select: CustomEvent<SelectData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type TableProps = typeof __propDef.props;
export type TableEvents = typeof __propDef.events;
export type TableSlots = typeof __propDef.slots;
export default class Table extends SvelteComponent<TableProps, TableEvents, TableSlots> {
}
export {};
