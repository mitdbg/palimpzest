import { SvelteComponent } from "svelte";
export { default as BaseRadio } from "./shared/Radio.svelte";
export { default as BaseExample } from "./Example.svelte";
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
        label?: any;
        info?: string | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: (string | null) | undefined;
        choices?: [string, string | number][] | undefined;
        show_label?: boolean | undefined;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        loading_status: LoadingStatus;
        interactive?: boolean | undefined;
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
