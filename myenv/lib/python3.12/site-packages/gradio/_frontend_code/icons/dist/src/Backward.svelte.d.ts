/** @typedef {typeof __propDef.props}  BackwardProps */
/** @typedef {typeof __propDef.events}  BackwardEvents */
/** @typedef {typeof __propDef.slots}  BackwardSlots */
export default class Backward extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type BackwardProps = typeof __propDef.props;
export type BackwardEvents = typeof __propDef.events;
export type BackwardSlots = typeof __propDef.slots;
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
