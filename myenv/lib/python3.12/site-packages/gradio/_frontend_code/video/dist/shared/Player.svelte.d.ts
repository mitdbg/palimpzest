import { SvelteComponent } from "svelte";
import type { FileData, Client } from "@gradio/client";
import type { I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        root?: string | undefined;
        src: string;
        subtitle?: (string | null) | undefined;
        mirror: boolean;
        autoplay: boolean;
        loop: boolean;
        label?: string | undefined;
        interactive?: boolean | undefined;
        handle_change?: ((video: FileData) => void) | undefined;
        handle_reset_value?: (() => void) | undefined;
        upload: Client["upload"];
        is_stream: boolean | undefined;
        i18n: I18nFormatter;
        show_download_button?: boolean | undefined;
        value?: FileData | null;
        handle_clear?: (() => void) | undefined;
        has_change_history?: boolean | undefined;
    };
    events: {
        play: CustomEvent<any>;
        pause: CustomEvent<any>;
        loadstart: Event;
        loadeddata: Event;
        loadedmetadata: Event;
        stop: CustomEvent<undefined>;
        end: CustomEvent<undefined>;
        clear: CustomEvent<undefined>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type PlayerProps = typeof __propDef.props;
export type PlayerEvents = typeof __propDef.events;
export type PlayerSlots = typeof __propDef.slots;
export default class Player extends SvelteComponent<PlayerProps, PlayerEvents, PlayerSlots> {
}
export {};
