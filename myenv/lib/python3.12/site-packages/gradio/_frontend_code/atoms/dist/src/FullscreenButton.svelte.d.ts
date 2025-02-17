import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        container?: HTMLElement | undefined;
    };
    events: {
        fullscreenchange: CustomEvent<boolean>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type FullscreenButtonProps = typeof __propDef.props;
export type FullscreenButtonEvents = typeof __propDef.events;
export type FullscreenButtonSlots = typeof __propDef.slots;
export default class FullscreenButton extends SvelteComponent<FullscreenButtonProps, FullscreenButtonEvents, FullscreenButtonSlots> {
}
export {};
