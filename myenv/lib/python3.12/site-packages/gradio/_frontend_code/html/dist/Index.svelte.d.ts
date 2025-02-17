import { SvelteComponent } from "svelte";
import type { Gradio } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        label?: string | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: string | undefined;
        loading_status: LoadingStatus;
        gradio: Gradio<{
            change: never;
            click: never;
            clear_status: LoadingStatus;
        }>;
        show_label?: boolean | undefined;
        min_height?: number | undefined;
        max_height?: number | undefined;
        container?: boolean | undefined;
        padding?: boolean | undefined;
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
