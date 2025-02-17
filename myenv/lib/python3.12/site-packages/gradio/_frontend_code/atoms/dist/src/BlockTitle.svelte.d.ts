import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        show_label?: boolean | undefined;
        info?: string | undefined;
        root: string;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type BlockTitleProps = typeof __propDef.props;
export type BlockTitleEvents = typeof __propDef.events;
export type BlockTitleSlots = typeof __propDef.slots;
export default class BlockTitle extends SvelteComponent<BlockTitleProps, BlockTitleEvents, BlockTitleSlots> {
}
export {};
