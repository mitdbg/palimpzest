import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        show_border?: boolean | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type ToolbarProps = typeof __propDef.props;
export type ToolbarEvents = typeof __propDef.events;
export type ToolbarSlots = typeof __propDef.slots;
export default class Toolbar extends SvelteComponent<ToolbarProps, ToolbarEvents, ToolbarSlots> {
}
export {};
