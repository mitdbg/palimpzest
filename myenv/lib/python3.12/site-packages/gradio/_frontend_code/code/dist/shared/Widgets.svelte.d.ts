import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        value: string;
        language: string;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type WidgetsProps = typeof __propDef.props;
export type WidgetsEvents = typeof __propDef.events;
export type WidgetsSlots = typeof __propDef.slots;
export default class Widgets extends SvelteComponent<WidgetsProps, WidgetsEvents, WidgetsSlots> {
}
export {};
