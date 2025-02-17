import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        size?: ("small" | "large") | undefined;
        unpadded_box?: boolean | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type EmptyProps = typeof __propDef.props;
export type EmptyEvents = typeof __propDef.events;
export type EmptySlots = typeof __propDef.slots;
export default class Empty extends SvelteComponent<EmptyProps, EmptyEvents, EmptySlots> {
}
export {};
