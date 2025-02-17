import { SvelteComponent } from "svelte";
import { type FileData } from "@gradio/client";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        variant?: ("primary" | "secondary" | "stop" | "huggingface") | undefined;
        size?: ("sm" | "md" | "lg") | undefined;
        value?: (string | null) | undefined;
        link?: (string | null) | undefined;
        icon?: FileData | null;
        disabled?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
    };
    events: {
        click: MouseEvent;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type ButtonProps = typeof __propDef.props;
export type ButtonEvents = typeof __propDef.events;
export type ButtonSlots = typeof __propDef.slots;
export default class Button extends SvelteComponent<ButtonProps, ButtonEvents, ButtonSlots> {
}
export {};
