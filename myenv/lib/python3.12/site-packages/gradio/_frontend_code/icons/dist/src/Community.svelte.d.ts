/** @typedef {typeof __propDef.props}  CommunityProps */
/** @typedef {typeof __propDef.events}  CommunityEvents */
/** @typedef {typeof __propDef.slots}  CommunitySlots */
export default class Community extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type CommunityProps = typeof __propDef.props;
export type CommunityEvents = typeof __propDef.events;
export type CommunitySlots = typeof __propDef.slots;
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
