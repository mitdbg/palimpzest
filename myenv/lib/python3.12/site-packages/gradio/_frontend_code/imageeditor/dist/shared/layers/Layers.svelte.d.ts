import { SvelteComponent } from "svelte";
import type { FileData } from "@gradio/client";
declare const __propDef: {
    props: {
        layer_files?: ((FileData | null)[] | null) | undefined;
        enable_layers?: boolean | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type LayersProps = typeof __propDef.props;
export type LayersEvents = typeof __propDef.events;
export type LayersSlots = typeof __propDef.slots;
export default class Layers extends SvelteComponent<LayersProps, LayersEvents, LayersSlots> {
}
export {};
