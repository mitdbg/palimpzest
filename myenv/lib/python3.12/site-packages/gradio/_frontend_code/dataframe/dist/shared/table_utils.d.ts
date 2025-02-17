import type { HeadersWithIDs } from "./utils";
export type TableData = {
    value: string | number;
    id: string;
}[][];
export declare function copy_table_data(data: TableData, headers?: HeadersWithIDs): Promise<void>;
