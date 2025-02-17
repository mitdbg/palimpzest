import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        elem_classes?: string[] | undefined;
        value: string;
        visible?: boolean | undefined;
    };
    events: {
        change: CustomEvent<undefined>;
        click: CustomEvent<undefined>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type HtmlProps = typeof __propDef.props;
export type HtmlEvents = typeof __propDef.events;
export type HtmlSlots = typeof __propDef.slots;
export default class Html extends SvelteComponent<HtmlProps, HtmlEvents, HtmlSlots> {
}
export {};
