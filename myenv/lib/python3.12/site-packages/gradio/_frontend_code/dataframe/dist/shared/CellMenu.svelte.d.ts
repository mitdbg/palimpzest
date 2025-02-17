import { SvelteComponent } from "svelte";
import type { I18nFormatter } from "js/utils/src";
declare const __propDef: {
    props: {
        x: number;
        y: number;
        on_add_row_above: () => void;
        on_add_row_below: () => void;
        on_add_column_left: () => void;
        on_add_column_right: () => void;
        row: number;
        col_count: [number, "fixed" | "dynamic"];
        row_count: [number, "fixed" | "dynamic"];
        on_delete_row: () => void;
        on_delete_col: () => void;
        can_delete_rows: boolean;
        can_delete_cols: boolean;
        i18n: I18nFormatter;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type CellMenuProps = typeof __propDef.props;
export type CellMenuEvents = typeof __propDef.events;
export type CellMenuSlots = typeof __propDef.slots;
export default class CellMenu extends SvelteComponent<CellMenuProps, CellMenuEvents, CellMenuSlots> {
}
export {};
