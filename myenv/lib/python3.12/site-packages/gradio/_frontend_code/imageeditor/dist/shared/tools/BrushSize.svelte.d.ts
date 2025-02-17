import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        selected_size: number;
        min: number;
        max: number;
    };
    events: {
        click_outside: CustomEvent<void>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type BrushSizeProps = typeof __propDef.props;
export type BrushSizeEvents = typeof __propDef.events;
export type BrushSizeSlots = typeof __propDef.slots;
export default class BrushSize extends SvelteComponent<BrushSizeProps, BrushSizeEvents, BrushSizeSlots> {
}
export {};
