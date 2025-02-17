import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        height?: number | string | undefined;
        min_height?: number | string | undefined;
        max_height?: number | string | undefined;
        width?: number | string | undefined;
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        variant?: ("solid" | "dashed" | "none") | undefined;
        border_mode?: ("base" | "focus" | "contrast") | undefined;
        padding?: boolean | undefined;
        type?: ("normal" | "fieldset") | undefined;
        test_id?: string | undefined;
        explicit_call?: boolean | undefined;
        container?: boolean | undefined;
        visible?: boolean | undefined;
        allow_overflow?: boolean | undefined;
        overflow_behavior?: ("visible" | "auto") | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        flex?: boolean | undefined;
        resizeable?: boolean | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type BlockProps = typeof __propDef.props;
export type BlockEvents = typeof __propDef.events;
export type BlockSlots = typeof __propDef.slots;
export default class Block extends SvelteComponent<BlockProps, BlockEvents, BlockSlots> {
}
export {};
