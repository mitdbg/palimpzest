/** @typedef {typeof __propDef.props}  PlotProps */
/** @typedef {typeof __propDef.events}  PlotEvents */
/** @typedef {typeof __propDef.slots}  PlotSlots */
export default class Plot extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type PlotProps = typeof __propDef.props;
export type PlotEvents = typeof __propDef.events;
export type PlotSlots = typeof __propDef.slots;
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
