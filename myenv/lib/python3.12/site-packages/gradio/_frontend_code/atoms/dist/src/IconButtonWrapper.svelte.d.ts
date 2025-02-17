/** @typedef {typeof __propDef.props}  IconButtonWrapperProps */
/** @typedef {typeof __propDef.events}  IconButtonWrapperEvents */
/** @typedef {typeof __propDef.slots}  IconButtonWrapperSlots */
export default class IconButtonWrapper extends SvelteComponent<{
    top_panel?: boolean | undefined;
    display_top_corner?: boolean | undefined;
}, {
    [evt: string]: CustomEvent<any>;
}, {
    default: {};
}> {
}
export type IconButtonWrapperProps = typeof __propDef.props;
export type IconButtonWrapperEvents = typeof __propDef.events;
export type IconButtonWrapperSlots = typeof __propDef.slots;
import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        top_panel?: boolean | undefined;
        display_top_corner?: boolean | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export {};
