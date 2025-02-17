import { SvelteComponent } from "svelte";
import type { ToastMessage } from "./types";
declare const __propDef: {
    props: {
        title?: string | undefined;
        message?: string | undefined;
        type: ToastMessage["type"];
        id: number;
        duration?: (number | null) | undefined;
        visible?: boolean | undefined;
    };
    events: {
        click: MouseEvent;
        keydown: KeyboardEvent;
        close: CustomEvent<any>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ToastContentProps = typeof __propDef.props;
export type ToastContentEvents = typeof __propDef.events;
export type ToastContentSlots = typeof __propDef.slots;
export default class ToastContent extends SvelteComponent<ToastContentProps, ToastContentEvents, ToastContentSlots> {
}
export {};
