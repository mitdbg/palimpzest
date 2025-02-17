import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        label?: (string | null) | undefined;
        Icon: any;
        show_label?: boolean | undefined;
        disable?: boolean | undefined;
        float?: boolean | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type BlockLabelProps = typeof __propDef.props;
export type BlockLabelEvents = typeof __propDef.events;
export type BlockLabelSlots = typeof __propDef.slots;
export default class BlockLabel extends SvelteComponent<BlockLabelProps, BlockLabelEvents, BlockLabelSlots> {
}
export {};
