/** @typedef {typeof __propDef.props}  ClearProps */
/** @typedef {typeof __propDef.events}  ClearEvents */
/** @typedef {typeof __propDef.slots}  ClearSlots */
export default class Clear extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type ClearProps = typeof __propDef.props;
export type ClearEvents = typeof __propDef.events;
export type ClearSlots = typeof __propDef.slots;
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
