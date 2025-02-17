import { SvelteComponent } from "svelte";
export { default as BaseChatBot } from "./shared/ChatBot.svelte";
import type { Gradio, SelectData, LikeData, CopyData } from "@gradio/utils";
import type { UndoRetryData } from "./shared/utils";
import type { LoadingStatus } from "@gradio/statustracker";
import type { FileData } from "@gradio/client";
import type { Message, ExampleMessage, TupleFormat } from "./types";
declare const __propDef: {
    props: {
        elem_id?: string | undefined;
        elem_classes?: string[] | undefined;
        visible?: boolean | undefined;
        value?: (TupleFormat | Message[]) | undefined;
        scale?: (number | null) | undefined;
        min_width?: number | undefined;
        label: string;
        show_label?: boolean | undefined;
        root: string;
        _selectable?: boolean | undefined;
        likeable?: boolean | undefined;
        feedback_options?: string[] | undefined;
        feedback_value?: ((string | null)[] | null) | undefined;
        show_share_button?: boolean | undefined;
        rtl?: boolean | undefined;
        show_copy_button?: boolean | undefined;
        show_copy_all_button?: boolean | undefined;
        sanitize_html?: boolean | undefined;
        layout?: ("bubble" | "panel") | undefined;
        type?: ("tuples" | "messages") | undefined;
        render_markdown?: boolean | undefined;
        line_breaks?: boolean | undefined;
        autoscroll?: boolean | undefined;
        _retryable?: boolean | undefined;
        _undoable?: boolean | undefined;
        group_consecutive_messages?: boolean | undefined;
        latex_delimiters: {
            left: string;
            right: string;
            display: boolean;
        }[];
        gradio: Gradio<{
            change: TupleFormat | Message[];
            select: SelectData;
            share: ShareData;
            error: string;
            like: LikeData;
            clear_status: LoadingStatus;
            example_select: SelectData;
            option_select: SelectData;
            edit: SelectData;
            retry: UndoRetryData;
            undo: UndoRetryData;
            clear: null;
            copy: CopyData;
        }>;
        avatar_images?: [FileData | null, FileData | null] | undefined;
        like_user_message?: boolean | undefined;
        loading_status?: LoadingStatus | undefined;
        height: number | string | undefined;
        resizeable: boolean;
        min_height: number | string | undefined;
        max_height: number | string | undefined;
        editable?: ("user" | "all" | null) | undefined;
        placeholder?: (string | null) | undefined;
        examples?: (ExampleMessage[] | null) | undefined;
        theme_mode: "system" | "light" | "dark";
        allow_file_downloads?: boolean | undefined;
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
