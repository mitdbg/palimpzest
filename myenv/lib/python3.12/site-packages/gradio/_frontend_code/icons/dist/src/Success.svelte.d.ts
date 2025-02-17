/** @typedef {typeof __propDef.props}  SuccessProps */
/** @typedef {typeof __propDef.events}  SuccessEvents */
/** @typedef {typeof __propDef.slots}  SuccessSlots */
export default class Success extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type SuccessProps = typeof __propDef.props;
export type SuccessEvents = typeof __propDef.events;
export type SuccessSlots = typeof __propDef.slots;
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
