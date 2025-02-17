import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        color?: string | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ColorPickerProps = typeof __propDef.props;
export type ColorPickerEvents = typeof __propDef.events;
export type ColorPickerSlots = typeof __propDef.slots;
export default class ColorPicker extends SvelteComponent<ColorPickerProps, ColorPickerEvents, ColorPickerSlots> {
}
export {};
