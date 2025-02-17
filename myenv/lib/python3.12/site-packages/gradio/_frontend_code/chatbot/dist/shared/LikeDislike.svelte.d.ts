import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        handle_action: (selected: string | null) => void;
        feedback_options: string[];
        selected?: (string | null) | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type LikeDislikeProps = typeof __propDef.props;
export type LikeDislikeEvents = typeof __propDef.events;
export type LikeDislikeSlots = typeof __propDef.slots;
export default class LikeDislike extends SvelteComponent<LikeDislikeProps, LikeDislikeEvents, LikeDislikeSlots> {
}
export {};
