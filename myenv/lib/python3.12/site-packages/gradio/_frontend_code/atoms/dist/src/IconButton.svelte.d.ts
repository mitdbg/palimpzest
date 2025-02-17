import { SvelteComponent } from "svelte";
import { type ComponentType } from "svelte";
declare const __propDef: {
    props: {
        Icon: ComponentType;
        label?: string | undefined;
        show_label?: boolean | undefined;
        pending?: boolean | undefined;
        size?: ("small" | "large" | "medium") | undefined;
        padded?: boolean | undefined;
        highlight?: boolean | undefined;
        disabled?: boolean | undefined;
        hasPopup?: boolean | undefined;
        color?: string | undefined;
        transparent?: boolean | undefined;
        background?: string | undefined;
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
export type IconButtonProps = typeof __propDef.props;
export type IconButtonEvents = typeof __propDef.events;
export type IconButtonSlots = typeof __propDef.slots;
export default class IconButton extends SvelteComponent<IconButtonProps, IconButtonEvents, IconButtonSlots> {
}
export {};
