import { SvelteComponent } from "svelte";
import type { FileData } from "@gradio/client";
declare const __propDef: {
    props: {
        value: FileData;
        zoom_speed: number;
        pan_speed: number;
        resolved_url?: any;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type Canvas3DgsProps = typeof __propDef.props;
export type Canvas3DgsEvents = typeof __propDef.events;
export type Canvas3DgsSlots = typeof __propDef.slots;
export default class Canvas3Dgs extends SvelteComponent<Canvas3DgsProps, Canvas3DgsEvents, Canvas3DgsSlots> {
}
export {};
