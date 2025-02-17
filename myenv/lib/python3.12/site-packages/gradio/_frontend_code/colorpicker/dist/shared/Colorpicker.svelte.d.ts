import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        value?: string | undefined;
        value_is_output?: boolean | undefined;
        label: string;
        info?: string | undefined;
        disabled?: boolean | undefined;
        show_label?: boolean | undefined;
        root: string;
        current_mode?: ("hex" | "rgb" | "hsl") | undefined;
        dialog_open?: boolean | undefined;
    };
    events: {
        focus: CustomEvent<any>;
        blur: CustomEvent<any>;
        change: CustomEvent<string>;
        click_outside: CustomEvent<void>;
        input: CustomEvent<undefined>;
        submit: CustomEvent<undefined>;
        selected: CustomEvent<string>;
        close: CustomEvent<void>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ColorpickerProps = typeof __propDef.props;
export type ColorpickerEvents = typeof __propDef.events;
export type ColorpickerSlots = typeof __propDef.slots;
export default class Colorpicker extends SvelteComponent<ColorpickerProps, ColorpickerEvents, ColorpickerSlots> {
}
export {};
