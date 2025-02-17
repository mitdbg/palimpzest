import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        icon: string;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type CellMenuIconsProps = typeof __propDef.props;
export type CellMenuIconsEvents = typeof __propDef.events;
export type CellMenuIconsSlots = typeof __propDef.slots;
export default class CellMenuIcons extends SvelteComponent<CellMenuIconsProps, CellMenuIconsEvents, CellMenuIconsSlots> {
}
export {};
