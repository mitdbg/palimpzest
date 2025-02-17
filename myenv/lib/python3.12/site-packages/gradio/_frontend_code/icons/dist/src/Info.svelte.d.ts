/** @typedef {typeof __propDef.props}  InfoProps */
/** @typedef {typeof __propDef.events}  InfoEvents */
/** @typedef {typeof __propDef.slots}  InfoSlots */
export default class Info extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type InfoProps = typeof __propDef.props;
export type InfoEvents = typeof __propDef.events;
export type InfoSlots = typeof __propDef.slots;
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
