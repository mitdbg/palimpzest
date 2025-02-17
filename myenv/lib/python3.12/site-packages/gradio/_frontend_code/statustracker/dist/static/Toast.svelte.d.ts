import { SvelteComponent } from "svelte";
import type { ToastMessage } from "./types";
declare const __propDef: {
    props: {
        messages?: ToastMessage[] | undefined;
    };
    events: {
        close: CustomEvent<any>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ToastProps = typeof __propDef.props;
export type ToastEvents = typeof __propDef.events;
export type ToastSlots = typeof __propDef.slots;
export default class Toast extends SvelteComponent<ToastProps, ToastEvents, ToastSlots> {
}
export {};
