/** @typedef {typeof __propDef.props}  TrashProps */
/** @typedef {typeof __propDef.events}  TrashEvents */
/** @typedef {typeof __propDef.slots}  TrashSlots */
export default class Trash extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type TrashProps = typeof __propDef.props;
export type TrashEvents = typeof __propDef.events;
export type TrashSlots = typeof __propDef.slots;
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
