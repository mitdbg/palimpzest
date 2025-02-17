import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        [x: string]: never;
    };
    events: {
        click: CustomEvent<undefined>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type WebcamPermissionsProps = typeof __propDef.props;
export type WebcamPermissionsEvents = typeof __propDef.events;
export type WebcamPermissionsSlots = typeof __propDef.slots;
export default class WebcamPermissions extends SvelteComponent<WebcamPermissionsProps, WebcamPermissionsEvents, WebcamPermissionsSlots> {
}
export {};
