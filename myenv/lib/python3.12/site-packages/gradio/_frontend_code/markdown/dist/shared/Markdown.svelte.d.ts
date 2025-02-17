import { SvelteComponent } from "svelte";
import type { CopyData } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value: string;
        min_height?: number | string | undefined;
        rtl?: boolean | undefined;
        sanitize_html?: boolean | undefined;
        line_breaks?: boolean | undefined;
        latex_delimiters: {
            left: string;
            right: string;
            display: boolean;
        }[];
        header_links?: boolean | undefined;
        height?: number | string | undefined;
        show_copy_button?: boolean | undefined;
        root: string;
        loading_status?: LoadingStatus | undefined;
    };
    events: {
        change: CustomEvent<undefined>;
        copy: CustomEvent<CopyData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type MarkdownProps = typeof __propDef.props;
export type MarkdownEvents = typeof __propDef.events;
export type MarkdownSlots = typeof __propDef.slots;
export default class Markdown extends SvelteComponent<MarkdownProps, MarkdownEvents, MarkdownSlots> {
}
export {};
