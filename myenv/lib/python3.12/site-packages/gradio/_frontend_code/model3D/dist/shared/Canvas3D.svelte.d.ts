import { SvelteComponent } from "svelte";
import type { FileData } from "@gradio/client";
declare const __propDef: {
    props: {
        value: FileData;
        display_mode: "solid" | "point_cloud" | "wireframe";
        clear_color: [number, number, number, number];
        camera_position: [number | null, number | null, number | null];
        zoom_speed: number;
        pan_speed: number;
        resolved_url?: any;
        reset_camera_position?: ((camera_position: [number | null, number | null, number | null], zoom_speed: number, pan_speed: number) => void) | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type Canvas3DProps = typeof __propDef.props;
export type Canvas3DEvents = typeof __propDef.events;
export type Canvas3DSlots = typeof __propDef.slots;
export default class Canvas3D extends SvelteComponent<Canvas3DProps, Canvas3DEvents, Canvas3DSlots> {
    get reset_camera_position(): (camera_position: [number | null, number | null, number | null], zoom_speed: number, pan_speed: number) => void;
}
export {};
