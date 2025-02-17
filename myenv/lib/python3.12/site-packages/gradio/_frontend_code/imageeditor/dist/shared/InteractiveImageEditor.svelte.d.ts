import { SvelteComponent } from "svelte";
export interface EditorData {
    background: FileData | null;
    layers: FileData[] | null;
    composite: FileData | null;
}
export interface ImageBlobs {
    background: FileData | null;
    layers: FileData[];
    composite: FileData | null;
}
import { type I18nFormatter } from "@gradio/utils";
import { type FileData, type Client } from "@gradio/client";
import { type CommandNode } from "./utils/commands";
import { type Brush as IBrush } from "./tools/Brush.svelte";
import { type Eraser } from "./tools/Brush.svelte";
declare const __propDef: {
    props: {
        brush: IBrush | null;
        eraser: Eraser | null;
        sources: ("clipboard" | "webcam" | "upload")[];
        crop_size?: ([number, number] | `${string}:${string}` | null) | undefined;
        i18n: I18nFormatter;
        root: string;
        label?: string | undefined;
        show_label: boolean;
        changeable?: boolean | undefined;
        value?: (EditorData | null) | undefined;
        transforms?: "crop"[] | undefined;
        layers: boolean;
        accept_blobs: (a: any) => void;
        status?: ("pending" | "complete" | "error" | "generating" | "streaming") | undefined;
        canvas_size: [number, number];
        fixed_canvas?: boolean | undefined;
        realtime: boolean;
        upload: Client["upload"];
        stream_handler: Client["stream"];
        dragging: boolean;
        placeholder?: string | undefined;
        dynamic_height?: number | undefined;
        height: any;
        full_history?: (CommandNode | null) | undefined;
        get_data?: (() => Promise<ImageBlobs>) | undefined;
        image_id?: (null | string) | undefined;
    };
    events: {
        history: CustomEvent<import("svelte/store").Writable<CommandNode>>;
        save: CustomEvent<void>;
        clear?: CustomEvent<undefined> | undefined;
        upload?: CustomEvent<undefined> | undefined;
        change?: CustomEvent<undefined> | undefined;
        receive_null?: CustomEvent<undefined> | undefined;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type InteractiveImageEditorProps = typeof __propDef.props;
export type InteractiveImageEditorEvents = typeof __propDef.events;
export type InteractiveImageEditorSlots = typeof __propDef.slots;
export default class InteractiveImageEditor extends SvelteComponent<InteractiveImageEditorProps, InteractiveImageEditorEvents, InteractiveImageEditorSlots> {
    get get_data(): () => Promise<ImageBlobs>;
}
export {};
