import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        [x: string]: never;
    };
    events: {
        remove_image: CustomEvent<any>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ClearImageProps = typeof __propDef.props;
export type ClearImageEvents = typeof __propDef.events;
export type ClearImageSlots = typeof __propDef.slots;
export default class ClearImage extends SvelteComponent<ClearImageProps, ClearImageEvents, ClearImageSlots> {
}
export {};
