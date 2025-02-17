import { SvelteComponent } from "svelte";
export { default as BaseTabs, TABS, type Tab } from "./shared/Tabs.svelte";
import type { Gradio, SelectData } from "@gradio/utils";
import { type Tab } from "./shared/Tabs.svelte";
declare const __propDef: {
    props: {
        visible?: boolean | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        selected: number | string;
        initial_tabs?: Tab[] | undefined;
        gradio: Gradio<{
            change: never;
            select: SelectData;
        }> | undefined;
    };
    events: {
        prop_change: CustomEvent<any>;
    } & {
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
