import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        color: string;
        current_mode?: ("hex" | "rgb" | "hsl") | undefined;
    };
    events: {
        selected: CustomEvent<string>;
        close: CustomEvent<void>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ColorFieldProps = typeof __propDef.props;
export type ColorFieldEvents = typeof __propDef.events;
export type ColorFieldSlots = typeof __propDef.slots;
export default class ColorField extends SvelteComponent<ColorFieldProps, ColorFieldEvents, ColorFieldSlots> {
}
export {};
