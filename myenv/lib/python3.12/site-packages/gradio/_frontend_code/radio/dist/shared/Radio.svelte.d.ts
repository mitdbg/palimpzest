import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        display_value: string;
        internal_value: string | number;
        disabled?: boolean | undefined;
        selected?: (string | null) | undefined;
    };
    events: {
        input: CustomEvent<string | number>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type RadioProps = typeof __propDef.props;
export type RadioEvents = typeof __propDef.events;
export type RadioSlots = typeof __propDef.slots;
export default class Radio extends SvelteComponent<RadioProps, RadioEvents, RadioSlots> {
}
export {};
