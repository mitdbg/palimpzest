/** @typedef {typeof __propDef.props}  RedoProps */
/** @typedef {typeof __propDef.events}  RedoEvents */
/** @typedef {typeof __propDef.slots}  RedoSlots */
export default class Redo extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type RedoProps = typeof __propDef.props;
export type RedoEvents = typeof __propDef.events;
export type RedoSlots = typeof __propDef.slots;
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
