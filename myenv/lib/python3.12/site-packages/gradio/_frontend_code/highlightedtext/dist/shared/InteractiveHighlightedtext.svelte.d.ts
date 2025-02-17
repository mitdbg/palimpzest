import { SvelteComponent } from "svelte";
import type { SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        value?: {
            token: string;
            class_or_confidence: string | number | null;
        }[] | undefined;
        show_legend?: boolean | undefined;
        color_map?: Record<string, string> | undefined;
        selectable?: boolean | undefined;
    };
    events: {
        select: CustomEvent<SelectData>;
        change: CustomEvent<{
            token: string;
            class_or_confidence: string | number | null;
        }[]>;
        input: CustomEvent<never>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type InteractiveHighlightedtextProps = typeof __propDef.props;
export type InteractiveHighlightedtextEvents = typeof __propDef.events;
export type InteractiveHighlightedtextSlots = typeof __propDef.slots;
export default class InteractiveHighlightedtext extends SvelteComponent<InteractiveHighlightedtextProps, InteractiveHighlightedtextEvents, InteractiveHighlightedtextSlots> {
}
export {};
