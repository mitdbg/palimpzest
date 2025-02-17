import { SvelteComponent } from "svelte";
import type { CopyData } from "@gradio/utils";
declare const __propDef: {
    props: {
        value: string;
    };
    events: {
        change: CustomEvent<undefined>;
        copy: CustomEvent<CopyData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type CopyProps = typeof __propDef.props;
export type CopyEvents = typeof __propDef.events;
export type CopySlots = typeof __propDef.slots;
export default class Copy extends SvelteComponent<CopyProps, CopyEvents, CopySlots> {
}
export {};
