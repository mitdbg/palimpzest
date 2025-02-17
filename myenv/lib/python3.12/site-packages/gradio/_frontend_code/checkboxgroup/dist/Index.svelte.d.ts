import { SvelteComponent } from "svelte";
import type { Gradio, SelectData } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        gradio: Gradio<{
            change: never;
            select: SelectData;
            input: never;
            clear_status: LoadingStatus;
        }>;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: (string | number)[] | undefined;
        choices: [string, string | number][];
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        label?: any;
        info?: string | undefined;
        show_label?: boolean | undefined;
        root: string;
        loading_status: LoadingStatus;
        interactive?: boolean | undefined;
        old_value?: (string | number)[] | undefined;
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
