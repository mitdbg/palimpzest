import { SvelteComponent } from "svelte";
import type { Gradio } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        gradio: Gradio<{
            change: never;
            input: never;
            submit: never;
            blur: never;
            focus: never;
            clear_status: LoadingStatus;
        }>;
        label?: any;
        info?: string | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        value?: number | undefined;
        show_label: boolean;
        minimum?: number | undefined;
        maximum?: number | undefined;
        loading_status: LoadingStatus;
        value_is_output?: boolean | undefined;
        step?: (number | null) | undefined;
        interactive: boolean;
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
}
export {};
