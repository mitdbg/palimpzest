import { SvelteComponent } from "svelte";
import type { Gradio, SelectData } from "@gradio/utils";
import type { LoadingStatus } from "@gradio/statustracker";
import { type FileData } from "@gradio/client";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: ({
            image: FileData;
            annotations: {
                image: FileData;
                label: string;
            }[] | [];
        } | null) | undefined;
        gradio: Gradio<{
            change: undefined;
            select: SelectData;
        }>;
        label?: any;
        show_label?: boolean | undefined;
        show_legend?: boolean | undefined;
        height: number | undefined;
        width: number | undefined;
        color_map: Record<string, string>;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        loading_status: LoadingStatus;
        show_fullscreen_button?: boolean | undefined;
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
export {};
