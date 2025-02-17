import { SvelteComponent } from "svelte";
import type { SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        label: string;
        id?: (string | number | object) | undefined;
        visible: boolean;
        interactive: boolean;
        order: number;
        scale: number;
    };
    events: {
        select: CustomEvent<SelectData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type TabItemProps = typeof __propDef.props;
export type TabItemEvents = typeof __propDef.events;
export type TabItemSlots = typeof __propDef.slots;
export default class TabItem extends SvelteComponent<TabItemProps, TabItemEvents, TabItemSlots> {
}
export {};
