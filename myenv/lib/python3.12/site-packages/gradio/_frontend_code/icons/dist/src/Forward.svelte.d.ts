/** @typedef {typeof __propDef.props}  ForwardProps */
/** @typedef {typeof __propDef.events}  ForwardEvents */
/** @typedef {typeof __propDef.slots}  ForwardSlots */
export default class Forward extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type ForwardProps = typeof __propDef.props;
export type ForwardEvents = typeof __propDef.events;
export type ForwardSlots = typeof __propDef.slots;
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
