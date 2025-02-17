import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        sources: Partial<"clipboard" | "upload" | "microphone" | "webcam" | null>[];
        active_source: Partial<"clipboard" | "upload" | "microphone" | "webcam" | null>;
        handle_clear?: (() => void) | undefined;
        handle_select?: ((source_type: Partial<"clipboard" | "upload" | "microphone" | "webcam" | null>) => void) | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type SelectSourceProps = typeof __propDef.props;
export type SelectSourceEvents = typeof __propDef.events;
export type SelectSourceSlots = typeof __propDef.slots;
export default class SelectSource extends SvelteComponent<SelectSourceProps, SelectSourceEvents, SelectSourceSlots> {
}
export {};
