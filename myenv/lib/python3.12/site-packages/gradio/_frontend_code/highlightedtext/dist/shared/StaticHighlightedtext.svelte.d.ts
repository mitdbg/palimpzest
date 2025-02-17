import { SvelteComponent } from "svelte";
import type { SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        value?: {
            token: string;
            class_or_confidence: string | number | null;
        }[] | undefined;
        show_legend?: boolean | undefined;
        show_inline_category?: boolean | undefined;
        color_map?: Record<string, string> | undefined;
        selectable?: boolean | undefined;
    };
    events: {
        select: CustomEvent<SelectData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type StaticHighlightedtextProps = typeof __propDef.props;
export type StaticHighlightedtextEvents = typeof __propDef.events;
export type StaticHighlightedtextSlots = typeof __propDef.slots;
export default class StaticHighlightedtext extends SvelteComponent<StaticHighlightedtextProps, StaticHighlightedtextEvents, StaticHighlightedtextSlots> {
}
export {};
