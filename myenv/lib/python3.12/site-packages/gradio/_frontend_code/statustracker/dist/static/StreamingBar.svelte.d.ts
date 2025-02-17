import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        time_limit: number | null;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type StreamingBarProps = typeof __propDef.props;
export type StreamingBarEvents = typeof __propDef.events;
export type StreamingBarSlots = typeof __propDef.slots;
export default class StreamingBar extends SvelteComponent<StreamingBarProps, StreamingBarEvents, StreamingBarSlots> {
}
export {};
