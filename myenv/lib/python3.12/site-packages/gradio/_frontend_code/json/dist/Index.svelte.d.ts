import { SvelteComponent } from "svelte";
export { default as BaseJSON } from "./shared/JSON.svelte";
import type { Gradio } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value: any;
        loading_status: LoadingStatus;
        label: string;
        show_label: boolean;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        gradio: Gradio<{
            change: never;
            clear_status: LoadingStatus;
        }>;
        open?: boolean | undefined;
        theme_mode: "system" | "light" | "dark";
        show_indices: boolean;
        height: number | string | undefined;
        min_height: number | string | undefined;
        max_height: number | string | undefined;
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
