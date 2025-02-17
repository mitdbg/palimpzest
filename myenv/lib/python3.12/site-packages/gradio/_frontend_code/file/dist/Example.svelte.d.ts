import { SvelteComponent } from "svelte";
import type { FileData } from "@gradio/client";
declare const __propDef: {
    props: {
        value: FileData | null;
        type: "gallery" | "table";
        selected?: boolean | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ExampleProps = typeof __propDef.props;
export type ExampleEvents = typeof __propDef.events;
export type ExampleSlots = typeof __propDef.slots;
export default class Example extends SvelteComponent<ExampleProps, ExampleEvents, ExampleSlots> {
}
export {};
