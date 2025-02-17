import { SvelteComponent } from "svelte";
export declare const TABS: {};
export interface Tab {
    label: string;
    id: string | number;
    elem_id: string | undefined;
    visible: boolean;
    interactive: boolean;
    scale: number | null;
}
import type { SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        visible?: boolean | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        selected: number | string;
        initial_tabs: Tab[];
    };
    events: {
        change: CustomEvent<undefined>;
        select: CustomEvent<SelectData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type TabsProps = typeof __propDef.props;
export type TabsEvents = typeof __propDef.events;
export type TabsSlots = typeof __propDef.slots;
export default class Tabs extends SvelteComponent<TabsProps, TabsEvents, TabsSlots> {
}
export {};
