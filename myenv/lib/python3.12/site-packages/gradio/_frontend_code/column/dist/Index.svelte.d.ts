import { SvelteComponent } from "svelte";
import type { LoadingStatus } from "@gradio/statustracker";
import type { Gradio } from "@gradio/utils";
declare const __propDef: {
    props: {
        scale?: (number | null) | undefined;
        gap?: boolean | undefined;
        min_width?: number | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        variant?: ("default" | "panel" | "compact") | undefined;
        loading_status?: LoadingStatus | undefined;
        gradio?: Gradio | undefined;
        show_progress?: boolean | undefined;
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
