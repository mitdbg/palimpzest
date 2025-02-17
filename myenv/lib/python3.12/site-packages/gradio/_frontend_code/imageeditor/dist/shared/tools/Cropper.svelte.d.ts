import { SvelteComponent } from "svelte";
import { type EditorContext } from "../ImageEditor.svelte";
declare const __propDef: {
    props: {
        editor_box: EditorContext["editor_box"];
        crop_constraint: number | null;
        w_p: number;
        h_p: number;
        l_p: number;
        t_p: number;
    };
    events: {
        crop_start: CustomEvent<{
            x: number;
            y: number;
            width: number;
            height: number;
        }>;
        crop_continue: CustomEvent<{
            x: number;
            y: number;
            width: number;
            height: number;
        }>;
        crop_end: CustomEvent<{
            x: number;
            y: number;
            width: number;
            height: number;
        }>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type CropperProps = typeof __propDef.props;
export type CropperEvents = typeof __propDef.events;
export type CropperSlots = typeof __propDef.slots;
export default class Cropper extends SvelteComponent<CropperProps, CropperEvents, CropperSlots> {
}
export {};
