/** @typedef {typeof __propDef.props}  BackProps */
/** @typedef {typeof __propDef.events}  BackEvents */
/** @typedef {typeof __propDef.slots}  BackSlots */
export default class Back extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type BackProps = typeof __propDef.props;
export type BackEvents = typeof __propDef.events;
export type BackSlots = typeof __propDef.slots;
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
