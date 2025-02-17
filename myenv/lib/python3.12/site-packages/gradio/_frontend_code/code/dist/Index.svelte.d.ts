import { SvelteComponent } from "svelte";
export { default as BaseCode } from "./shared/Code.svelte";
export { default as BaseCopy } from "./shared/Copy.svelte";
export { default as BaseDownload } from "./shared/Download.svelte";
export { default as BaseWidget } from "./shared/Widgets.svelte";
export { default as BaseExample } from "./Example.svelte";
import type { Gradio } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        gradio: Gradio<{
            change: string;
            input: never;
            blur: never;
            focus: never;
            clear_status: LoadingStatus;
        }>;
        value?: string | undefined;
        value_is_output?: boolean | undefined;
        language?: string | undefined;
        lines?: number | undefined;
        max_lines?: number | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        label?: any;
        show_label?: boolean | undefined;
        loading_status: LoadingStatus;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        wrap_lines?: boolean | undefined;
        interactive: boolean;
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
