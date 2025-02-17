import { SvelteComponent } from "svelte";
export { default as BasePlot } from "./shared/Plot.svelte";
import type { Gradio, SelectData } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        value?: (null | string) | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        loading_status: LoadingStatus;
        label: string;
        show_label: boolean;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        theme_mode: "system" | "light" | "dark";
        caption: string;
        bokeh_version: string | null;
        gradio: Gradio<{
            change: never;
            clear_status: LoadingStatus;
            select: SelectData;
        }>;
        show_actions_button?: boolean | undefined;
        _selectable?: boolean | undefined;
        x_lim?: ([number, number] | null) | undefined;
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
