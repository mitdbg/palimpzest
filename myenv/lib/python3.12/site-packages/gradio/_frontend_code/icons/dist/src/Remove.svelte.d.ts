/** @typedef {typeof __propDef.props}  RemoveProps */
/** @typedef {typeof __propDef.events}  RemoveEvents */
/** @typedef {typeof __propDef.slots}  RemoveSlots */
export default class Remove extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type RemoveProps = typeof __propDef.props;
export type RemoveEvents = typeof __propDef.events;
export type RemoveSlots = typeof __propDef.slots;
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
