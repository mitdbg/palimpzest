/** @typedef {typeof __propDef.props}  ErrorProps */
/** @typedef {typeof __propDef.events}  ErrorEvents */
/** @typedef {typeof __propDef.slots}  ErrorSlots */
export default class Error extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type ErrorProps = typeof __propDef.props;
export type ErrorEvents = typeof __propDef.events;
export type ErrorSlots = typeof __propDef.slots;
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
