import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        crop_constraint: [number, number] | `${string}:${string}` | null;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type CropProps = typeof __propDef.props;
export type CropEvents = typeof __propDef.events;
export type CropSlots = typeof __propDef.slots;
export default class Crop extends SvelteComponent<CropProps, CropEvents, CropSlots> {
}
export {};
