import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        currentVolume: number;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type VolumeLevelsProps = typeof __propDef.props;
export type VolumeLevelsEvents = typeof __propDef.events;
export type VolumeLevelsSlots = typeof __propDef.slots;
export default class VolumeLevels extends SvelteComponent<VolumeLevelsProps, VolumeLevelsEvents, VolumeLevelsSlots> {
}
export {};
