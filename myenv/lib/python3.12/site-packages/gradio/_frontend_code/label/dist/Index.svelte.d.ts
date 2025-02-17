import { SvelteComponent } from "svelte";
export { default as BaseLabel } from "./shared/Label.svelte";
import type { Gradio, SelectData } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
declare const __propDef: {
    props: {
        gradio: Gradio<{
            change: never;
            select: SelectData;
            clear_status: LoadingStatus;
        }>;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        color?: undefined | string;
        value?: {
            label?: string;
            confidences?: {
                label: string;
                confidence: number;
            }[];
        } | undefined;
        label?: any;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        loading_status: LoadingStatus;
        show_label?: boolean | undefined;
        _selectable?: boolean | undefined;
        show_heading?: boolean | undefined;
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
