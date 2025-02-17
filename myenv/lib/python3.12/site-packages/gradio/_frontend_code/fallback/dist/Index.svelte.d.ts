import { SvelteComponent } from "svelte";
import type { Gradio } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
import type { SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: boolean | undefined;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        loading_status: LoadingStatus;
        gradio: Gradio<{
            change: never;
            select: SelectData;
            input: never;
            clear_status: LoadingStatus;
        }>;
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
