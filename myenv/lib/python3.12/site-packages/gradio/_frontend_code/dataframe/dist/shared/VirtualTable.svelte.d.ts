import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        items?: any[][] | undefined;
        max_height: number;
        actual_height: number;
        table_scrollbar_width: number;
        start?: number | undefined;
        end?: number | undefined;
        selected: number | false;
        scroll_to_index?: ((index: number, opts: ScrollToOptions, align_end?: boolean) => Promise<void>) | undefined;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        thead: {};
        tbody: {
            item: any[];
            index: number;
        };
        tfoot: {};
    };
};
export type VirtualTableProps = typeof __propDef.props;
export type VirtualTableEvents = typeof __propDef.events;
export type VirtualTableSlots = typeof __propDef.slots;
export default class VirtualTable extends SvelteComponent<VirtualTableProps, VirtualTableEvents, VirtualTableSlots> {
    get scroll_to_index(): (index: number, opts: ScrollToOptions, align_end?: boolean) => Promise<void>;
}
export {};
