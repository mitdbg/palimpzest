import { SvelteComponent } from "svelte";
import type { HTMLVideoAttributes } from "svelte/elements";
declare const __propDef: {
    props: {
        [x: string]: any;
        src?: HTMLVideoAttributes["src"];
        muted?: HTMLVideoAttributes["muted"];
        playsinline?: HTMLVideoAttributes["playsinline"];
        preload?: HTMLVideoAttributes["preload"];
        autoplay?: HTMLVideoAttributes["autoplay"];
        controls?: HTMLVideoAttributes["controls"];
        currentTime?: number | undefined;
        duration?: number | undefined;
        paused?: boolean | undefined;
        node?: HTMLVideoElement | undefined;
        loop: boolean;
        is_stream: any;
        processingVideo?: boolean | undefined;
    };
    events: {
        loadstart: Event;
        loadeddata: Event;
        loadedmetadata: Event;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type VideoProps = typeof __propDef.props;
export type VideoEvents = typeof __propDef.events;
export type VideoSlots = typeof __propDef.slots;
export default class Video extends SvelteComponent<VideoProps, VideoEvents, VideoSlots> {
}
export {};
