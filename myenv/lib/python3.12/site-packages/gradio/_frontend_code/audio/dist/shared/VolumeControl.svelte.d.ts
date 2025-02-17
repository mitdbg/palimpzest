import { SvelteComponent } from "svelte";
import WaveSurfer from "wavesurfer.js";
declare const __propDef: {
    props: {
        currentVolume?: number | undefined;
        show_volume_slider?: boolean | undefined;
        waveform: WaveSurfer | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type VolumeControlProps = typeof __propDef.props;
export type VolumeControlEvents = typeof __propDef.events;
export type VolumeControlSlots = typeof __propDef.slots;
export default class VolumeControl extends SvelteComponent<VolumeControlProps, VolumeControlEvents, VolumeControlSlots> {
}
export {};
