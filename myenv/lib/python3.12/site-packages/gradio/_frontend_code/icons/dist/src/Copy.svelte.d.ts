/** @typedef {typeof __propDef.props}  CopyProps */
/** @typedef {typeof __propDef.events}  CopyEvents */
/** @typedef {typeof __propDef.slots}  CopySlots */
export default class Copy extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type CopyProps = typeof __propDef.props;
export type CopyEvents = typeof __propDef.events;
export type CopySlots = typeof __propDef.slots;
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
