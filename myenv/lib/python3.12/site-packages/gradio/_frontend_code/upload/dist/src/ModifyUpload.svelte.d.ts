import { SvelteComponent } from "svelte";
import type { I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        editable?: boolean | undefined;
        undoable?: boolean | undefined;
        download?: (string | null) | undefined;
        i18n: I18nFormatter;
    };
    events: {
        edit?: CustomEvent<undefined> | undefined;
        clear?: CustomEvent<undefined> | undefined;
        undo?: CustomEvent<undefined> | undefined;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type ModifyUploadProps = typeof __propDef.props;
export type ModifyUploadEvents = typeof __propDef.events;
export type ModifyUploadSlots = typeof __propDef.slots;
export default class ModifyUpload extends SvelteComponent<ModifyUploadProps, ModifyUploadEvents, ModifyUploadSlots> {
}
export {};
