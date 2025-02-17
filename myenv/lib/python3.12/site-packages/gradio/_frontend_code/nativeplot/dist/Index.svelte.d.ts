import { SvelteComponent } from "svelte";
import type { Gradio, SelectData } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        value: {
            columns: string[];
            data: [string | number][];
            datatypes: Record<string, "quantitative" | "temporal" | "nominal">;
            mark: "line" | "point" | "bar";
        } | null;
        x: string;
        y: string;
        color?: (string | null) | undefined;
        root: string;
        title?: (string | null) | undefined;
        x_title?: (string | null) | undefined;
        y_title?: (string | null) | undefined;
        color_title?: (string | null) | undefined;
        x_bin?: (string | number | null) | undefined;
        y_aggregate?: "sum" | "mean" | "median" | "min" | "max" | undefined;
        color_map?: (Record<string, string> | null) | undefined;
        x_lim?: ([number, number] | null) | undefined;
        y_lim?: ([number, number] | null) | undefined;
        x_label_angle?: (number | null) | undefined;
        y_label_angle?: (number | null) | undefined;
        x_axis_labels_visible?: boolean | undefined;
        caption?: (string | null) | undefined;
        sort?: ("x" | "y" | "-x" | "-y" | string[] | null) | undefined;
        tooltip?: ("axis" | "none" | "all" | string[]) | undefined;
        _selectable?: boolean | undefined;
        gradio: Gradio<{
            select: SelectData;
            double_click: undefined;
            clear_status: LoadingStatus;
        }>;
        label?: string | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        show_label: boolean;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        loading_status?: LoadingStatus | undefined;
        height?: number | undefined;
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
}
export {};
