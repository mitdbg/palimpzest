/** @typedef {typeof __propDef.props}  ChatProps */
/** @typedef {typeof __propDef.events}  ChatEvents */
/** @typedef {typeof __propDef.slots}  ChatSlots */
export default class Chat extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type ChatProps = typeof __propDef.props;
export type ChatEvents = typeof __propDef.events;
export type ChatSlots = typeof __propDef.slots;
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
