/** @typedef {typeof __propDef.props}  MusicProps */
/** @typedef {typeof __propDef.events}  MusicEvents */
/** @typedef {typeof __propDef.slots}  MusicSlots */
export default class Music extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type MusicProps = typeof __propDef.props;
export type MusicEvents = typeof __propDef.events;
export type MusicSlots = typeof __propDef.slots;
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
