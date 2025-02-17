import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        value: Record<string, {
            type: string;
            description: string;
            default: string;
        }>;
        linkify?: string[] | undefined;
        header?: (string | null) | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type IndexProps = typeof __propDef.props;
export type IndexEvents = typeof __propDef.events;
export type IndexSlots = typeof __propDef.slots;
export default class Index extends SvelteComponent<IndexProps, IndexEvents, IndexSlots> {
}
export {};
