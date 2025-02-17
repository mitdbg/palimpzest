import { SvelteComponent } from "svelte";
import type { LoadingStatus } from "@gradio/statustracker";
import type { Gradio } from "@gradio/utils";
declare const __propDef: {
    props: {
        label: string;
        elem_id: string;
        elem_classes: string[];
        visible?: boolean | undefined;
        open?: boolean | undefined;
        loading_status: LoadingStatus;
        gradio: Gradio<{
            expand: never;
            collapse: never;
        }>;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type IndexProps = typeof __propDef.props;
export type IndexEvents = typeof __propDef.events;
export type IndexSlots = typeof __propDef.slots;
export default class Index extends SvelteComponent<IndexProps, IndexEvents, IndexSlots> {
}
export {};
