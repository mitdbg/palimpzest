/** @typedef {typeof __propDef.props}  CalendarProps */
/** @typedef {typeof __propDef.events}  CalendarEvents */
/** @typedef {typeof __propDef.slots}  CalendarSlots */
export default class Calendar extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type CalendarProps = typeof __propDef.props;
export type CalendarEvents = typeof __propDef.events;
export type CalendarSlots = typeof __propDef.slots;
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
