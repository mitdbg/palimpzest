/** @typedef {typeof __propDef.props}  EraseProps */
/** @typedef {typeof __propDef.events}  EraseEvents */
/** @typedef {typeof __propDef.slots}  EraseSlots */
export default class Erase extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type EraseProps = typeof __propDef.props;
export type EraseEvents = typeof __propDef.events;
export type EraseSlots = typeof __propDef.slots;
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
