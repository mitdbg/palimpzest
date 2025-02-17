import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        info: string;
        root: string;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type InfoProps = typeof __propDef.props;
export type InfoEvents = typeof __propDef.events;
export type InfoSlots = typeof __propDef.slots;
export default class Info extends SvelteComponent<InfoProps, InfoEvents, InfoSlots> {
}
export {};
