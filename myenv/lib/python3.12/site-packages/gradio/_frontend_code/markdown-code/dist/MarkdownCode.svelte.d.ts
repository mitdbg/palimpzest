import { SvelteComponent } from "svelte";
import "katex/dist/katex.min.css";
import "./prism.css";
declare const __propDef: {
    props: {
        chatbot?: boolean | undefined;
        message: string;
        sanitize_html?: boolean | undefined;
        latex_delimiters?: {
            left: string;
            right: string;
            display: boolean;
        }[] | undefined;
        render_markdown?: boolean | undefined;
        line_breaks?: boolean | undefined;
        header_links?: boolean | undefined;
        root: string;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type MarkdownCodeProps = typeof __propDef.props;
export type MarkdownCodeEvents = typeof __propDef.events;
export type MarkdownCodeSlots = typeof __propDef.slots;
export default class MarkdownCode extends SvelteComponent<MarkdownCodeProps, MarkdownCodeEvents, MarkdownCodeSlots> {
}
export {};
