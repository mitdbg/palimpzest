import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        value: any;
        depth?: number | undefined;
        is_root?: boolean | undefined;
        is_last_item?: boolean | undefined;
        key?: (string | number | null) | undefined;
        open?: boolean | undefined;
        theme_mode?: ("system" | "light" | "dark") | undefined;
        show_indices?: boolean | undefined;
    };
    events: {
        toggle: CustomEvent<any>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type JsonNodeProps = typeof __propDef.props;
export type JsonNodeEvents = typeof __propDef.events;
export type JsonNodeSlots = typeof __propDef.slots;
export default class JsonNode extends SvelteComponent<JsonNodeProps, JsonNodeEvents, JsonNodeSlots> {
}
export {};
