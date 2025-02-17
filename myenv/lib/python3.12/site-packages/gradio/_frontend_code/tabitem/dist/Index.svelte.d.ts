import { SvelteComponent } from "svelte";
export { default as BaseTabItem } from "./shared/TabItem.svelte";
import type { Gradio, SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        label: string;
        id: string | number;
        gradio: Gradio<{
            select: SelectData;
        }> | undefined;
        visible?: boolean | undefined;
        interactive?: boolean | undefined;
        order: number;
        scale: number;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type IndexProps = typeof __propDef.props;
export type IndexEvents = typeof __propDef.events;
export type IndexSlots = typeof __propDef.slots;
export default class Index extends SvelteComponent<IndexProps, IndexEvents, IndexSlots> {
}
