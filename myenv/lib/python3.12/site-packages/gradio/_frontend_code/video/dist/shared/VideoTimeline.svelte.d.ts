import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        videoElement: HTMLVideoElement;
        trimmedDuration: number | null;
        dragStart: number;
        dragEnd: number;
        loadingTimeline: boolean;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type VideoTimelineProps = typeof __propDef.props;
export type VideoTimelineEvents = typeof __propDef.events;
export type VideoTimelineSlots = typeof __propDef.slots;
export default class VideoTimeline extends SvelteComponent<VideoTimelineProps, VideoTimelineEvents, VideoTimelineSlots> {
}
export {};
