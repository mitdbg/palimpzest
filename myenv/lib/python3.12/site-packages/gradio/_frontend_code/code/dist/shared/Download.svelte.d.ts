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
export type DownloadProps = typeof __propDef.props;
export type DownloadEvents = typeof __propDef.events;
export type DownloadSlots = typeof __propDef.slots;
export default class Download extends SvelteComponent<DownloadProps, DownloadEvents, DownloadSlots> {
}
export {};
