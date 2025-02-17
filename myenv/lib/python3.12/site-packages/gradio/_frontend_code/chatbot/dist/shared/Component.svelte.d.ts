import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        type: "gallery" | "plot" | "audio" | "video" | "image" | "dataframe" | string;
        components: any;
        value: any;
        target: any;
        theme_mode: any;
        props: any;
        i18n: any;
        upload: any;
        _fetch: any;
        allow_file_downloads: boolean;
        display_icon_button_wrapper_top_corner?: boolean | undefined;
    };
    events: {
        load: any;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ComponentProps = typeof __propDef.props;
export type ComponentEvents = typeof __propDef.events;
export type ComponentSlots = typeof __propDef.slots;
export default class Component extends SvelteComponent<ComponentProps, ComponentEvents, ComponentSlots> {
}
export {};
