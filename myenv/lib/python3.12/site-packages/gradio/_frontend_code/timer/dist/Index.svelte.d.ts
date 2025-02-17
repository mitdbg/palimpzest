import { SvelteComponent } from "svelte";
import type { Gradio } from "@gradio/utils";
declare const __propDef: {
    props: {
        gradio: Gradio<{
            tick: never;
        }>;
        value?: number | undefined;
        active?: boolean | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type IndexProps = typeof __propDef.props;
export type IndexEvents = typeof __propDef.events;
export type IndexSlots = typeof __propDef.slots;
export default class Index extends SvelteComponent<IndexProps, IndexEvents, IndexSlots> {
}
export {};
