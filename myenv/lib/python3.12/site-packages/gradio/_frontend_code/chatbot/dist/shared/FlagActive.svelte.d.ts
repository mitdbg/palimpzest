/** @typedef {typeof __propDef.props}  FlagActiveProps */
/** @typedef {typeof __propDef.events}  FlagActiveEvents */
/** @typedef {typeof __propDef.slots}  FlagActiveSlots */
export default class FlagActive extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type FlagActiveProps = typeof __propDef.props;
export type FlagActiveEvents = typeof __propDef.events;
export type FlagActiveSlots = typeof __propDef.slots;
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
