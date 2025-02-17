import { SvelteComponent } from "svelte";
export { default as BaseStaticHighlightedText } from "./shared/StaticHighlightedtext.svelte";
export { default as BaseInteractiveHighlightedText } from "./shared/InteractiveHighlightedtext.svelte";
import type { Gradio, SelectData } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        gradio: Gradio<{
            select: SelectData;
            change: never;
            clear_status: LoadingStatus;
        }>;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value: {
            token: string;
            class_or_confidence: string | number | null;
        }[];
        show_legend: boolean;
        show_inline_category: boolean;
        color_map?: Record<string, string> | undefined;
        label?: any;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        _selectable?: boolean | undefined;
        combine_adjacent?: boolean | undefined;
        interactive: boolean;
        show_label?: boolean | undefined;
        loading_status: LoadingStatus;
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
