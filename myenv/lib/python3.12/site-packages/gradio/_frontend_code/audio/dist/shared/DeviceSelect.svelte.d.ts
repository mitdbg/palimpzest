import { SvelteComponent } from "svelte";
import type { I18nFormatter } from "@gradio/utils";
declare const __propDef: {
    props: {
        i18n: I18nFormatter;
        micDevices?: MediaDeviceInfo[] | undefined;
    };
    events: {
        error: CustomEvent<string>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type DeviceSelectProps = typeof __propDef.props;
export type DeviceSelectEvents = typeof __propDef.events;
export type DeviceSelectSlots = typeof __propDef.slots;
export default class DeviceSelect extends SvelteComponent<DeviceSelectProps, DeviceSelectEvents, DeviceSelectSlots> {
}
export {};
