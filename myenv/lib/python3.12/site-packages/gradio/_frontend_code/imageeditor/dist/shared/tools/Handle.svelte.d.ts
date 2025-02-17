import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        type?: string | undefined;
        location: "tl" | "tr" | "bl" | "br" | "t" | "b" | "l" | "r";
        x1?: number | undefined;
        y1?: number | undefined;
        x2?: number | undefined;
        y2?: number | undefined;
        dragging?: boolean | undefined;
    };
    events: {
        change: CustomEvent<{
            top: number | undefined;
            bottom: number | undefined;
            left: number | undefined;
            right: number | undefined;
        }>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type HandleProps = typeof __propDef.props;
export type HandleEvents = typeof __propDef.events;
export type HandleSlots = typeof __propDef.slots;
export default class Handle extends SvelteComponent<HandleProps, HandleEvents, HandleSlots> {
}
export {};
