import { SvelteComponent } from "svelte";
export { default as BaseMarkdown } from "./shared/Markdown.svelte";
export { default as BaseExample } from "./Example.svelte";
import type { Gradio, CopyData } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        label: string;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: string | undefined;
        loading_status: LoadingStatus;
        rtl?: boolean | undefined;
        sanitize_html?: boolean | undefined;
        line_breaks?: boolean | undefined;
        gradio: Gradio<{
            change: never;
            copy: CopyData;
            clear_status: LoadingStatus;
        }>;
        latex_delimiters: {
            left: string;
            right: string;
            display: boolean;
        }[];
        header_links?: boolean | undefined;
        height: number | string | undefined;
        min_height: number | string | undefined;
        max_height: number | string | undefined;
        show_copy_button?: boolean | undefined;
        container?: boolean | undefined;
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
