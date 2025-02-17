import { SvelteComponent } from "svelte";
import type { FileData, Client } from "@gradio/client";
import type { I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        value: null | FileData;
        display_mode?: ("solid" | "point_cloud" | "wireframe") | undefined;
        clear_color?: [number, number, number, number] | undefined;
        label?: string | undefined;
        show_label: boolean;
        root: string;
        i18n: I18nFormatter;
        zoom_speed?: number | undefined;
        pan_speed?: number | undefined;
        max_file_size?: (number | null) | undefined;
        uploading?: boolean | undefined;
        camera_position?: [number | null, number | null, number | null] | undefined;
        upload: Client["upload"];
        stream_handler: Client["stream"];
    };
    events: {
        error: CustomEvent<any>;
        change: CustomEvent<any>;
        clear: CustomEvent<undefined>;
        drag: CustomEvent<boolean>;
        load: CustomEvent<FileData>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type Model3DUploadProps = typeof __propDef.props;
export type Model3DUploadEvents = typeof __propDef.events;
export type Model3DUploadSlots = typeof __propDef.slots;
export default class Model3DUpload extends SvelteComponent<Model3DUploadProps, Model3DUploadEvents, Model3DUploadSlots> {
}
export {};
