/** @typedef {typeof __propDef.props}  SketchProps */
/** @typedef {typeof __propDef.events}  SketchEvents */
/** @typedef {typeof __propDef.slots}  SketchSlots */
export default class Sketch extends SvelteComponent<{
    [x: string]: never;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type SketchProps = typeof __propDef.props;
export type SketchEvents = typeof __propDef.events;
export type SketchSlots = typeof __propDef.slots;
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
