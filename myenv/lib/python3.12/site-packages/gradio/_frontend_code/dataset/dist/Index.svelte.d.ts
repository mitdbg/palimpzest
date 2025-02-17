import { SvelteComponent } from "svelte";
import type { ComponentType } from "svelte";
import type { Gradio, SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        components: string[];
        component_props: Record<string, any>[];
        component_map: Map<string, Promise<{
            default: ComponentType<SvelteComponent>;
        }>>;
        label?: string | undefined;
        show_label?: boolean | undefined;
        headers: string[];
        samples?: (any[][] | null) | undefined;
        sample_labels?: (string[] | null) | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: (number | null) | undefined;
        root: string;
        proxy_url: null | string;
        samples_per_page?: number | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        gradio: Gradio<{
            click: number;
            select: SelectData;
        }>;
        layout?: ("gallery" | "table" | null) | undefined;
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
