import { SvelteComponent } from "svelte";
export { default as BaseExample } from "./Example.svelte";
import type { Gradio } from "@gradio/utils";
declare const __propDef: {
    props: {
        gradio: Gradio<{
            change: undefined;
            submit: undefined;
        }>;
        label?: string | undefined;
        show_label?: boolean | undefined;
        info?: string | undefined;
        interactive: boolean;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: string | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        root: string;
        include_time?: boolean | undefined;
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
