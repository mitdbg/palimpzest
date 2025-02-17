/** @typedef {typeof __propDef.props}  WarningProps */
/** @typedef {typeof __propDef.events}  WarningEvents */
/** @typedef {typeof __propDef.slots}  WarningSlots */
export default class Warning extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type WarningProps = typeof __propDef.props;
export type WarningEvents = typeof __propDef.events;
export type WarningSlots = typeof __propDef.slots;
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
