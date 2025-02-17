/** @typedef {typeof __propDef.props}  UndoProps */
/** @typedef {typeof __propDef.events}  UndoEvents */
/** @typedef {typeof __propDef.slots}  UndoSlots */
export default class Undo extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type UndoProps = typeof __propDef.props;
export type UndoEvents = typeof __propDef.events;
export type UndoSlots = typeof __propDef.slots;
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
