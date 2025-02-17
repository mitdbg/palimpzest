import { SvelteComponent } from "svelte";
import type { NormalisedMessage } from "../types";
declare const __propDef: {
    props: {
        value: NormalisedMessage[] | null;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type CopyAllProps = typeof __propDef.props;
export type CopyAllEvents = typeof __propDef.events;
export type CopyAllSlots = typeof __propDef.slots;
export default class CopyAll extends SvelteComponent<CopyAllProps, CopyAllEvents, CopyAllSlots> {
}
export {};
