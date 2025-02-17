import { SvelteComponent } from "svelte";
import type { FileData } from "@gradio/client";
import type { I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        value: FileData | null;
        display_mode?: ("solid" | "point_cloud" | "wireframe") | undefined;
        clear_color?: [number, number, number, number] | undefined;
        label?: string | undefined;
        show_label: boolean;
        i18n: I18nFormatter;
        zoom_speed?: number | undefined;
        pan_speed?: number | undefined;
        camera_position?: [number | null, number | null, number | null] | undefined;
        has_change_history?: boolean | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type Model3DProps = typeof __propDef.props;
export type Model3DEvents = typeof __propDef.events;
export type Model3DSlots = typeof __propDef.slots;
export default class Model3D extends SvelteComponent<Model3DProps, Model3DEvents, Model3DSlots> {
}
export {};
