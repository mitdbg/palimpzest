import { SvelteComponent } from "svelte";
import type { Gradio } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        gradio: Gradio<{
            change: never;
            input: never;
            release: number;
            clear_status: LoadingStatus;
        }>;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: number | undefined;
        label?: any;
        info?: string | undefined;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        minimum: number;
        maximum?: number | undefined;
        step: number;
        show_label: boolean;
        interactive: boolean;
        loading_status: LoadingStatus;
        value_is_output?: boolean | undefined;
        root: string;
        show_reset_button: boolean;
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
