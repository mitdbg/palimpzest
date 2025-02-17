/** @typedef {typeof __propDef.props}  SquareProps */
/** @typedef {typeof __propDef.events}  SquareEvents */
/** @typedef {typeof __propDef.slots}  SquareSlots */
export default class Square extends SvelteComponent<{
    fill?: string | undefined;
    stroke_width?: number | undefined;
}, {
    [evt: string]: CustomEvent<any>;
}, {}> {
}
export type SquareProps = typeof __propDef.props;
export type SquareEvents = typeof __propDef.events;
export type SquareSlots = typeof __propDef.slots;
import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        fill?: string | undefined;
        stroke_width?: number | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export {};
