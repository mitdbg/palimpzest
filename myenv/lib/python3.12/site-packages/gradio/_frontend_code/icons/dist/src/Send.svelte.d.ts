/** @typedef {typeof __propDef.props}  SendProps */
/** @typedef {typeof __propDef.events}  SendEvents */
/** @typedef {typeof __propDef.slots}  SendSlots */
export default class Send extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type SendProps = typeof __propDef.props;
export type SendEvents = typeof __propDef.events;
export type SendSlots = typeof __propDef.slots;
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
