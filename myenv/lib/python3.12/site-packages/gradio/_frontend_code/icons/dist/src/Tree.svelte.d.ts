/** @typedef {typeof __propDef.props}  TreeProps */
/** @typedef {typeof __propDef.events}  TreeEvents */
/** @typedef {typeof __propDef.slots}  TreeSlots */
export default class Tree extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type TreeProps = typeof __propDef.props;
export type TreeEvents = typeof __propDef.events;
export type TreeSlots = typeof __propDef.slots;
import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        [x: string]: never;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export {};
