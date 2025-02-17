import { SvelteComponent } from "svelte";
export { default as BaseModel3D } from "./shared/Model3D.svelte";
export { default as BaseModel3DUpload } from "./shared/Model3DUpload.svelte";
export { default as BaseExample } from "./Example.svelte";
import type { FileData } from "@gradio/client";
import type { LoadingStatus } from "@gradio/statustracker";
import type { Gradio } from "@gradio/utils";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: null | FileData;
        root: string;
        display_mode?: ("solid" | "point_cloud" | "wireframe") | undefined;
        clear_color: [number, number, number, number];
        loading_status: LoadingStatus;
        label: string;
        show_label: boolean;
        container?: boolean | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        gradio: Gradio;
        height?: number | undefined;
        zoom_speed?: number | undefined;
        input_ready: boolean;
        has_change_history?: boolean | undefined;
        camera_position?: [number | null, number | null, number | null] | undefined;
        interactive: boolean;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type IndexProps = typeof __propDef.props;
export type IndexEvents = typeof __propDef.events;
export type IndexSlots = typeof __propDef.slots;
export default class Index extends SvelteComponent<IndexProps, IndexEvents, IndexSlots> {
}
