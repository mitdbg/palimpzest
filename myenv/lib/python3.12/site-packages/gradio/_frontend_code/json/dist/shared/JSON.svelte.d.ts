import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        value?: any;
        open?: boolean | undefined;
        theme_mode?: ("system" | "light" | "dark") | undefined;
        show_indices?: boolean | undefined;
        label_height: number;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type JsonProps = typeof __propDef.props;
export type JsonEvents = typeof __propDef.events;
export type JsonSlots = typeof __propDef.slots;
export default class Json extends SvelteComponent<JsonProps, JsonEvents, JsonSlots> {
}
export {};
