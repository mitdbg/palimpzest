import { SvelteComponent } from "svelte";
import type { ShareData } from "@gradio/utils";
import type { I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        formatter: (arg0: any) => Promise<string>;
        value: any;
        i18n: I18nFormatter;
    };
    events: {
        share: CustomEvent<ShareData>;
        error: CustomEvent<string>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ShareButtonProps = typeof __propDef.props;
export type ShareButtonEvents = typeof __propDef.events;
export type ShareButtonSlots = typeof __propDef.slots;
export default class ShareButton extends SvelteComponent<ShareButtonProps, ShareButtonEvents, ShareButtonSlots> {
}
export {};
