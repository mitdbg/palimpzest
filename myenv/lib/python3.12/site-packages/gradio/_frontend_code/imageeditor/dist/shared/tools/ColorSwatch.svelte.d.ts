import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        selected_color: string;
        colors: string[];
        user_colors?: ((string | null)[] | null) | undefined;
        show_empty?: boolean | undefined;
        current_mode?: ("hex" | "rgb" | "hsl") | undefined;
        color_picker?: boolean | undefined;
    };
    events: {
        select: CustomEvent<{
            index: number | null;
            color: string | null;
        }>;
        edit: CustomEvent<{
            index: number;
            color: string | null;
        }>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ColorSwatchProps = typeof __propDef.props;
export type ColorSwatchEvents = typeof __propDef.events;
export type ColorSwatchSlots = typeof __propDef.slots;
export default class ColorSwatch extends SvelteComponent<ColorSwatchProps, ColorSwatchEvents, ColorSwatchSlots> {
}
export {};
