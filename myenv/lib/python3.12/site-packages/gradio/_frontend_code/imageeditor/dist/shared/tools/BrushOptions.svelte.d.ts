import { SvelteComponent } from "svelte";
import { type Brush } from "./Brush.svelte";
declare const __propDef: {
    props: {
        colors: string[];
        selected_color: string;
        color_mode?: Brush["color_mode"] | undefined;
        recent_colors?: (string | null)[] | undefined;
        selected_size: number;
        dimensions: [number, number];
        parent_width: number;
        parent_height: number;
        parent_left: number;
        toolbar_box: DOMRect | Record<string, never>;
        show_swatch: boolean;
    };
    events: {
        click_outside: CustomEvent<void>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type BrushOptionsProps = typeof __propDef.props;
export type BrushOptionsEvents = typeof __propDef.events;
export type BrushOptionsSlots = typeof __propDef.slots;
export default class BrushOptions extends SvelteComponent<BrushOptionsProps, BrushOptionsEvents, BrushOptionsSlots> {
}
export {};
