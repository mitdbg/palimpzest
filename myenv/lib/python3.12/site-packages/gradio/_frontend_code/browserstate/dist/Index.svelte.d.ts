import { SvelteComponent } from "svelte";
import type { Gradio } from "@gradio/utils";
declare const __propDef: {
    props: {
        storage_key: string;
        secret: string;
        default_value: any;
        value?: any;
        gradio: Gradio<{
            change: never;
        }>;
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
    get storage_key(): string;
    /**accessor*/
    set storage_key(_: string);
    get secret(): string;
    /**accessor*/
    set secret(_: string);
    get default_value(): any;
    /**accessor*/
    set default_value(_: any);
    get value(): any;
    /**accessor*/
    set value(_: any);
    get gradio(): Gradio<{
        change: never;
    }>;
    /**accessor*/
    set gradio(_: Gradio<{
        change: never;
    }>);
}
export {};
